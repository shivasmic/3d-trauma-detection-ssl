"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl

Modified V-DETR transformer components for 3D UNet features
IMPORTANT: Preserves 8-corner RPE mechanism (key innovation of V-DETR)

"""

from typing import Optional
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.helpers import ACTIVATION_DICT, NORM_DICT, get_clones
from models.vdetr_transformer import (
    BoxProcessor, inverse_sigmoid, convert_corners_camera2lidar,
    ShareSelfAttention, FFNLayer, roty_batch_tensor
)


class UNetCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, 
                 attn_drop=0.0, proj_drop=0.0, args=None):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.log_scale = getattr(args, 'log_scale', 512.0) if args else 512.0
        self.rpe_quant = getattr(args, 'rpe_quant', 'bilinear_4_10') if args else 'bilinear_4_10'
        self.angle_type = getattr(args, 'angle_type', '') if args else ''
        
        self.interp_method, max_value, num_points = self.rpe_quant.split('_')
        max_value, num_points = float(max_value), int(num_points)
        
        relative_coords_table = torch.stack(torch.meshgrid(
            torch.linspace(-max_value, max_value, num_points, dtype=torch.float32),
            torch.linspace(-max_value, max_value, num_points, dtype=torch.float32),
            torch.linspace(-max_value, max_value, num_points, dtype=torch.float32),
            indexing='ij'
        ), dim=-1).unsqueeze(0)
        self.register_buffer("relative_coords_table", relative_coords_table)
        self.max_value = max_value
        
        rpe_dim = getattr(args, 'rpe_dim', 128) if args else 128
        self.cpb_mlps = get_clones(self.build_cpb_mlp(3, rpe_dim, num_heads), 8)
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim // self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim, dim // self.num_heads, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, query, key, reference_point, reference_angle, 
                xyz_vol, attn_mask=None, key_padding_mask=None):
        
        key = key.permute(1, 0, 2)  
        query = query.permute(1, 0, 2)  
        
        B, nQ = reference_point.shape[:2]
        nK = xyz_vol.shape[1]
        
        
        rpe = None
        for i in range(8):  
            deltas = reference_point[:, :, None, i, :] - xyz_vol[:, None, :, :]  
            
            if self.angle_type == "object_coords" and reference_angle is not None:
                deltas[..., 2] *= -1
                deltas[..., [0, 1, 2]] = deltas[..., [0, 2, 1]]
                
                R = roty_batch_tensor(reference_angle)  
                deltas = torch.matmul(deltas, R)
                
                deltas[..., 1] *= -1
                deltas[..., [0, 1, 2]] = deltas[..., [0, 2, 1]] 
            
            deltas = torch.sign(deltas) * torch.log2(
                torch.abs(deltas) * self.log_scale + 1.0
            ) / np.log2(8)
            delta = deltas / self.max_value 
            
            rpe_table = self.cpb_mlps[i](self.relative_coords_table).permute(0, 4, 1, 2, 3)
            
            rpe_i = F.grid_sample(
                rpe_table,
                delta.view(1, 1, 1, -1, 3).to(rpe_table.dtype),
                mode=self.interp_method,
                align_corners=True if self.interp_method != 'nearest' else None
            )
            rpe_i = rpe_i.squeeze().view(-1, B, nQ, nK).permute(1, 0, 2, 3)
            
            if i == 0:
                rpe = rpe_i
            else:
                rpe = rpe + rpe_i 
        
        B_, N_key, C = key.shape
        k = self.k(key).reshape(B_, N_key, 1, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(key).reshape(B_, N_key, 1, C // self.num_heads).permute(0, 2, 1, 3)
        
        B_, N_query, C = query.shape
        q = self.q(query).reshape(B_, N_query, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        
        attn = q @ k.transpose(-2, -1) 
        attn = attn + rpe
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float(-100))
            else:
                attn = attn + attn_mask
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  
        
        x = x.transpose(1, 2).reshape(B_, N_query, C)
        x = self.proj(x)
        x = self.proj_drop(x).permute(1, 0, 2)  
        
        return x, attn


class UNetDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None, activation="relu",
                 normalize_before=True, norm_fn_name="ln",
                 pos_for_key=False, args=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.pos_for_key = pos_for_key
        
        if args is not None and hasattr(args, 'share_selfattn') and args.share_selfattn:
            self.self_attn = ShareSelfAttention(d_model, nhead, dropout=dropout, args=args)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.multihead_attn = UNetCrossAttention(
            d_model, nhead, attn_drop=dropout, proj_drop=dropout, args=args
        )
        
        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)
        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, reference_point, reference_angle,
                     xyz_vol, point_cloud_dims=None,
                     tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None, return_attn_weights=False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        if self.pos_for_key:
            tgt2, attn = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                reference_point=reference_point,  
                reference_angle=reference_angle,
                xyz_vol=xyz_vol,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        else:
            tgt2, attn = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=memory,
                reference_point=reference_point, 
                reference_angle=reference_angle,
                xyz_vol=xyz_vol,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt, memory, reference_point, reference_angle,
                    xyz_vol, point_cloud_dims=None,
                    tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None, return_attn_weights=False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        if self.pos_for_key:
            tgt2, attn = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                reference_point=reference_point, 
                reference_angle=reference_angle,
                xyz_vol=xyz_vol,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        else:
            tgt2, attn = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=memory,
                reference_point=reference_point,  
                reference_angle=reference_angle,
                xyz_vol=xyz_vol,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
        
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(self, tgt, memory, reference_point, reference_angle,
                xyz_vol, point_cloud_dims=None,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None, return_attn_weights=False):
       
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, reference_point, reference_angle, xyz_vol,
                point_cloud_dims, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                pos, query_pos, return_attn_weights
            )
        return self.forward_post(
            tgt, memory, reference_point, reference_angle, xyz_vol,
            point_cloud_dims, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask,
            pos, query_pos, return_attn_weights
        )
