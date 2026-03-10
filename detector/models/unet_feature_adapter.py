"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.position_embedding import PositionEmbeddingCoordsSine


class UNetFeatureAdapter(nn.Module):
    def __init__(self, 
                 unet_channels,
                 transformer_dim=256,
                 feature_resolution=None,
                 voxel_spacing=(1.0, 1.0, 1.0),
                 use_all_voxels=False,
                 max_voxels=8192,
                 use_fpn=False,
                 args=None):
       
        super().__init__()
        self.unet_channels = unet_channels
        self.transformer_dim = transformer_dim
        self.feature_resolution = feature_resolution
        self.voxel_spacing = torch.tensor(voxel_spacing)
        self.use_all_voxels = use_all_voxels
        self.max_voxels = max_voxels
        self.use_fpn = use_fpn
        
        self.feature_projection = nn.Sequential(
            nn.Conv3d(unet_channels, transformer_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, transformer_dim),
            nn.ReLU(inplace=True),
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(transformer_dim, transformer_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, transformer_dim),
            nn.ReLU(inplace=True),
        )
        
        self.pos_embed_3d = PositionEmbeddingCoordsSine(
            d_pos=transformer_dim,
            pos_type="fourier",
            normalize=True,
            gauss_scale=1.0
        )
        
        if feature_resolution is not None:
            n_voxels = feature_resolution[0] * feature_resolution[1] * feature_resolution[2]
            if n_voxels <= max_voxels:
                self.pos_embed_learned = nn.Parameter(
                    torch.randn(1, n_voxels, transformer_dim) * 0.02
                )
            else:
                self.pos_embed_learned = None
        else:
            self.pos_embed_learned = None
    
    def generate_voxel_coordinates(self, feature_shape, device, volume_dims=None):
        B, C, D, H, W = feature_shape
        
        if volume_dims is None:
            d_coords = torch.linspace(-1, 1, D, device=device)
            h_coords = torch.linspace(-1, 1, H, device=device)
            w_coords = torch.linspace(-1, 1, W, device=device)
        else:
            min_coords, max_coords = volume_dims
            d_coords = torch.linspace(min_coords[0].item(), max_coords[0].item(), D, device=device)
            h_coords = torch.linspace(min_coords[1].item(), max_coords[1].item(), H, device=device)
            w_coords = torch.linspace(min_coords[2].item(), max_coords[2].item(), W, device=device)
        
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        xyz = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        xyz = xyz.reshape(-1, 3)
        xyz = xyz.unsqueeze(0).expand(B, -1, -1)
        
        return xyz
    
    def sample_voxels_uniformly(self, features, xyz, n_samples):
        N, B, C = features.shape
        
        if N <= n_samples:
            return features, xyz
        
        step = N // n_samples
        indices = torch.arange(0, N, step, device=features.device)[:n_samples]
        
        sampled_features = features[indices] 
        sampled_xyz = xyz[:, indices, :]  
        
        return sampled_features, sampled_xyz
    
    def forward(self, unet_features, volume_dims=None):
        B, C, D, H, W = unet_features.shape
        device = unet_features.device
        
        if not self.voxel_spacing.is_cuda:
            self.voxel_spacing = self.voxel_spacing.to(device)
        
        features = self.feature_projection(unet_features) 
        features = self.spatial_conv(features) 
        xyz = self.generate_voxel_coordinates(features.shape, device, volume_dims)
        
        N_voxels = D * H * W
        features_flat = features.flatten(2)  
        features_flat = features_flat.permute(2, 0, 1)  
        
        if not self.use_all_voxels and N_voxels > self.max_voxels:
            features_flat, xyz = self.sample_voxels_uniformly(
                features_flat, xyz, self.max_voxels
            )
        

        if volume_dims is not None:
            pos_embed = self.pos_embed_3d(
                xyz, 
                num_channels=self.transformer_dim,
                input_range=volume_dims
            )  
        else:
            pos_embed = self.pos_embed_3d(
                xyz,
                num_channels=self.transformer_dim,
                input_range=None
            )  
        
        pos_embed = pos_embed.permute(2, 0, 1)  
        
        if self.pos_embed_learned is not None and features_flat.shape[0] == self.pos_embed_learned.shape[1]:
            learned_pos = self.pos_embed_learned.permute(1, 0, 2) 
            learned_pos = learned_pos.expand(-1, B, -1)  
            pos_embed = pos_embed + learned_pos
        
        return features_flat, xyz, pos_embed


class MultiScaleUNetAdapter(nn.Module):
    def __init__(self,
                 unet_channels_list,  
                 transformer_dim=256,
                 selected_scale=-1,  
                 args=None):
        super().__init__()
        self.selected_scale = selected_scale
        self.num_scales = len(unet_channels_list)
        
        self.scale_adapters = nn.ModuleList([
            UNetFeatureAdapter(
                unet_channels=ch,
                transformer_dim=transformer_dim,
                args=args
            )
            for ch in unet_channels_list
        ])
        
        self.use_multiscale = args.use_multiscale if args is not None else False
        if self.use_multiscale:
            self.scale_fusion = nn.Linear(
                transformer_dim * self.num_scales,
                transformer_dim
            )
    
    def forward(self, unet_features_list, volume_dims=None):
        if not self.use_multiscale:
            features, xyz, pos_embed = self.scale_adapters[self.selected_scale](
                unet_features_list[self.selected_scale],
                volume_dims
            )
            return features, xyz, pos_embed
        else:
            all_features = []
            for i, unet_feat in enumerate(unet_features_list):
                feat, xyz, pos = self.scale_adapters[i](unet_feat, volume_dims)
                all_features.append(feat)
            
            features_cat = torch.cat(all_features, dim=-1)  
            features = self.scale_fusion(features_cat)  
            
            _, xyz, pos_embed = self.scale_adapters[self.selected_scale](
                unet_features_list[self.selected_scale],
                volume_dims
            )
            
            return features, xyz, pos_embed
