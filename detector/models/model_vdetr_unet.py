# Copyright (c) Modified for UNet integration
"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl

Main V-DETR model adapted for 3D UNet encoder instead of sparse point cloud backbone.
Integrates pre-trained 3D UNet encoder with V-DETR transformer decoder.

"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from models.rsna_unet_adapter import RSNAUNetFeatureAdapter
from models.unet_transformer import UNetDecoderLayer, UNetCrossAttention, FFNLayer
from models.vdetr_transformer import TransformerDecoder, BoxProcessor
from models.helpers import GenericMLP, PositionEmbeddingLearned
from models.position_embedding import PositionEmbeddingCoordsSine


class ModelVDETR_UNet(nn.Module):
    def __init__(self,
                 unet_encoder,
                 decoder,
                 dataset_config,
                 unet_channels=256,  
                 unet_feature_resolution=(32, 21, 21),  
                 encoder_dim=256,
                 decoder_dim=256,
                 num_queries=256,  
                 voxel_spacing=(2.0, 1.0, 1.0), 
                 max_voxels=4096, 
                 querypos_mlp=False,
                 freeze_unet=False,
                 args=None):
        super().__init__()
        
        self.unet_encoder = unet_encoder
        self.freeze_unet = freeze_unet
        if freeze_unet:
            self._freeze_unet()
        
        self.unet_channels = unet_channels
        self.unet_feature_resolution = unet_feature_resolution
        self.voxel_spacing = voxel_spacing
        self.max_voxels = max_voxels
        
        self.feature_adapter = RSNAUNetFeatureAdapter(
            unet_channels=unet_channels,
            transformer_dim=encoder_dim,
            feature_resolution=unet_feature_resolution,
            use_all_voxels=False,
            max_voxels=max_voxels,
            args=args
        )
        
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=[encoder_dim],
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        
        self.querypos_mlp = querypos_mlp
        if not querypos_mlp:
            self.pos_embedding = PositionEmbeddingCoordsSine(
                d_pos=decoder_dim,
                pos_type="fourier",
                normalize=True
            )
            self.query_projection = GenericMLP(
                input_dim=decoder_dim,
                hidden_dims=[decoder_dim],
                output_dim=decoder_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        
        self.decoder = decoder
        self.num_queries = num_queries
        self.dataset_config = dataset_config
        
        self.use_learned_anchors = getattr(args, 'use_learned_anchors', False) if args else False
        if self.use_learned_anchors:
            self.anchor_centers = nn.Parameter(torch.randn(num_queries, 3) * 0.5)
            self.anchor_sizes = nn.Parameter(torch.ones(num_queries, 3) * 0.1)
    
    def _freeze_unet(self):
        for param in self.unet_encoder.parameters():
            param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"UNet encoder FROZEN")
        print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def _unfreeze_unet(self):
        for param in self.unet_encoder.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"UNet encoder UNFROZEN")
        print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def run_unet_encoder(self, dicom_volumes):
       
        with torch.set_grad_enabled(not self.freeze_unet):
            unet_features = self.unet_encoder.encode(dicom_volumes)
        
        
        features, xyz, pos_embed = self.feature_adapter(unet_features, volume_dims=None)
        features = self.encoder_to_decoder_projection(
            features.permute(1, 2, 0) 
        ).permute(2, 0, 1)  
        
        return features, xyz, pos_embed
    
    
    def get_query_embeddings(self, encoder_xyz, enc_features, volume_dims):
        B = encoder_xyz.shape[0]
        N = encoder_xyz.shape[1]  
        
        if N >= self.num_queries:
            step = N // self.num_queries
            indices = torch.arange(0, N, step, device=encoder_xyz.device)[:self.num_queries]
            query_xyz = encoder_xyz[:, indices, :] 
            query_inds = indices
        else:
            query_xyz = torch.zeros(B, self.num_queries, 3, device=encoder_xyz.device)
            query_xyz[:, :N, :] = encoder_xyz
            query_inds = None
        
        if not self.querypos_mlp:
            pos_embed = self.pos_embedding(query_xyz, input_range=volume_dims)
            query_embed = self.query_projection(pos_embed).permute(2, 0, 1)
        else:
            query_embed = query_xyz.permute(1, 0, 2)  
        
        
        return query_xyz, query_embed, query_inds
    
    def generate_initial_box_predictions(self, query_xyz, volume_dims, device):
        device = query_xyz.device
        if volume_dims is not None:
            volume_dims = [x.to(device) for x in volume_dims]

        B, nQ = query_xyz.shape[:2]

        
        if self.use_learned_anchors:
            center_unnorm = self.anchor_centers.unsqueeze(0).expand(B, -1, -1)
            size_unnorm = self.anchor_sizes.unsqueeze(0).expand(B, -1, -1)
        else:
            center_unnorm = query_xyz
            scene_size = volume_dims[1] - volume_dims[0]
            size_unnorm = scene_size.unsqueeze(1).expand(-1, nQ, -1) * 0.1
        
        scene_size = volume_dims[1] - volume_dims[0]
        center_norm = (center_unnorm - volume_dims[0].unsqueeze(1)) / scene_size.unsqueeze(1)
        size_norm = size_unnorm / scene_size.unsqueeze(1)
        
        half_size = size_unnorm / 2
        box_corners = torch.stack([
            center_unnorm - half_size,
            center_unnorm + half_size * torch.tensor([1, -1, -1], device=device),
            center_unnorm + half_size * torch.tensor([1, 1, -1], device=device),
            center_unnorm + half_size * torch.tensor([-1, 1, -1], device=device),
            center_unnorm + half_size * torch.tensor([-1, -1, 1], device=device),
            center_unnorm + half_size * torch.tensor([1, -1, 1], device=device),
            center_unnorm + half_size,
            center_unnorm + half_size * torch.tensor([-1, 1, 1], device=device),
        ], dim=2)  
        
        num_classes = self.dataset_config.num_semcls
        point_cls_logits = torch.zeros(B, nQ, num_classes, device=device)

        enc_box_predictions = {
            'center_normalized': center_norm,
            'center_unnormalized': center_unnorm,
            'size_normalized': size_norm,
            'size_unnormalized': size_unnorm,
            'box_corners': box_corners,
            'point_cls_logits': point_cls_logits,
        }
        
        return enc_box_predictions
    
    def forward(self, inputs, encoder_only=False):
        dicom_volumes = inputs['dicom_volumes']
        B, _, D, H, W = dicom_volumes.shape
        device = dicom_volumes.device
       

        if 'volume_dims' in inputs and inputs['volume_dims'] is not None:
            volume_dims = inputs['volume_dims'] 
            if isinstance(volume_dims, tuple):
                volume_dims = list(volume_dims)
            volume_dims = [x.to(device) for x in volume_dims]
        else:
            volume_dims = [
                torch.zeros(B, 3, device=device),
                torch.tensor([1024.0, 336.0, 336.0], device=device).unsqueeze(0).expand(B, -1)
            ]

        enc_features, enc_xyz, enc_pos = self.run_unet_encoder(dicom_volumes)


        if encoder_only:
            return {
                'enc_features': enc_features,
                'enc_xyz': enc_xyz,
                'enc_pos': enc_pos
            }
        
        query_xyz, query_embed, query_inds = self.get_query_embeddings(
            enc_xyz, enc_features, volume_dims
        )
        
        enc_box_predictions = self.generate_initial_box_predictions(
            query_xyz, volume_dims, device
        )
        
        if not self.querypos_mlp:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt = None
        
        enc_box_features = enc_features

        if query_inds is not None:
            enc_box_features = enc_features[query_inds, :, :]  
        else:
            enc_box_features = torch.zeros(self.num_queries, B, enc_features.shape[-1], device=enc_features.device)
            enc_box_features[:enc_features.shape[0], :, :] = enc_features


        box_predictions = self.decoder(
            tgt=tgt,
            memory=enc_features,
            query_xyz=query_xyz,
            enc_xyz=enc_xyz,  
            point_cloud_dims=volume_dims,
            query_pos=query_embed,
            enc_box_predictions=enc_box_predictions,
            enc_box_features=enc_box_features,
        )[0]
        
        box_predictions['enc_outputs'] = enc_box_predictions
        box_predictions['seed_xyz'] = enc_xyz
        
        return box_predictions


def build_unet_decoder(args, dataset_config):
    """Build transformer decoder with UNet-compatible layers"""
    from models.helpers import get_clones
    
    first_layer = FFNLayer(
        d_model=args.dec_dim,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
        norm_fn_name=args.dec_norm if hasattr(args, 'dec_norm') else 'ln',
    )
    
    decoder_layer = UNetDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
        pos_for_key=args.pos_for_key,
        norm_fn_name=args.dec_norm if hasattr(args, 'dec_norm') else 'ln',
        args=args
    )
    
    decoder = TransformerDecoder(
        first_layer=first_layer,
        decoder_layer=decoder_layer,
        dataset_config=dataset_config,
        num_layers=args.dec_nlayers - 1,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        mlp_norm=args.mlp_norm,
        mlp_act=args.mlp_act,
        mlp_sep=args.mlp_sep,
        pos_for_key=args.pos_for_key,
        num_queries=args.nqueries,
        cls_loss=args.cls_loss,
        is_bilable=args.is_bilable,
        q_content=args.q_content,
        return_intermediate=True,
        args=args
    )
    
    return decoder


def build_vdetr_unet(args, dataset_config, unet_encoder):
    decoder = build_unet_decoder(args, dataset_config)
    
    unet_channels = args.unet_channels  
    unet_resolution = tuple(args.unet_resolution)  
    voxel_spacing = (2.0, 1.0, 1.0)  
    
    model = ModelVDETR_UNet(
        unet_encoder=unet_encoder,
        decoder=decoder,
        dataset_config=dataset_config,
        unet_channels=unet_channels,
        unet_feature_resolution=unet_resolution,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        num_queries=args.nqueries,
        voxel_spacing=voxel_spacing,
        max_voxels=args.max_voxels if hasattr(args, 'max_voxels') else 4096,
        querypos_mlp=args.querypos_mlp,
        freeze_unet=args.freeze_unet if hasattr(args, 'freeze_unet') else False,
        args=args,
    )
    
    return model
