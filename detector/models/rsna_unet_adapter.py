"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.position_embedding import PositionEmbeddingCoordsSine
from dataset.coordinate_adapter import get_rsna_coordinate_adapter


class RSNAUNetFeatureAdapter(nn.Module):
    def __init__(self, 
                 unet_channels=256,  
                 transformer_dim=256,
                 feature_resolution=(32, 21, 21),  
                 use_all_voxels=False,
                 max_voxels=4096,  
                 args=None):
    
        super().__init__()
        self.unet_channels = unet_channels
        self.transformer_dim = transformer_dim
        self.feature_resolution = feature_resolution
        self.use_all_voxels = use_all_voxels
        self.max_voxels = max_voxels
        
        self.coord_adapter = get_rsna_coordinate_adapter()
        
        if feature_resolution is not None:
            self.downsample_factor = (
                512 // feature_resolution[0],  
                336 // feature_resolution[1],  
                336 // feature_resolution[2]   
            )
            
            
            self.effective_spacing = tuple(
                self.coord_adapter.voxel_spacing[i] * self.downsample_factor[i]
                for i in range(3)
            )
            
        else:
            self.downsample_factor = None
            self.effective_spacing = self.coord_adapter.voxel_spacing
        
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
    
    def generate_voxel_coordinates(self, feature_shape, device):
        B, C, D, H, W = feature_shape
        
        d_indices = torch.arange(D, device=device, dtype=torch.float32)
        h_indices = torch.arange(H, device=device, dtype=torch.float32)
        w_indices = torch.arange(W, device=device, dtype=torch.float32)
        
        spacing = torch.tensor(self.effective_spacing, device=device, dtype=torch.float32)
        
        d_coords = d_indices * spacing[0]  
        h_coords = h_indices * spacing[1]  
        w_coords = w_indices * spacing[2]  
        
        grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        xyz = torch.stack([grid_d, grid_h, grid_w], dim=-1)
        xyz = xyz.reshape(-1, 3)
        
        xyz = xyz.unsqueeze(0).expand(B, -1, -1)
        
        return xyz
    
    def sample_voxels_randomly(self, features, xyz, n_samples):
        N, B, C = features.shape
        
        if N <= n_samples:
            return features, xyz
        
       
        indices = torch.randperm(N, device=features.device)[:n_samples]
        indices = indices.sort()[0]  
        
        sampled_features = features[indices]  
        sampled_xyz = xyz[:, indices, :]  
        
        return sampled_features, sampled_xyz
    
    def forward(self, unet_features, volume_dims=None):
        B, C, D, H, W = unet_features.shape
        device = unet_features.device
        
        assert C == self.unet_channels, f"Expected {self.unet_channels} channels, got {C}"
        assert (D, H, W) == self.feature_resolution, \
            f"Expected resolution {self.feature_resolution}, got ({D}, {H}, {W})"
        
        features = self.feature_projection(unet_features) 
        features = self.spatial_conv(features)  
        xyz = self.generate_voxel_coordinates(features.shape, device)  
        
        N_voxels = D * H * W 
        features_flat = features.flatten(2)  
        features_flat = features_flat.permute(2, 0, 1)  
        
        if not self.use_all_voxels and N_voxels > self.max_voxels:
            features_flat, xyz = self.sample_voxels_randomly(
                features_flat, xyz, self.max_voxels
            )
        
        if volume_dims is None:
            min_coords = torch.zeros(B, 3, device=device)
            max_coords = torch.tensor(
                self.coord_adapter.physical_dims, 
                device=device
            ).unsqueeze(0).expand(B, -1)
            volume_dims = (min_coords, max_coords)
        
        pos_embed = self.pos_embed_3d(
            xyz, 
            num_channels=self.transformer_dim,
            input_range=volume_dims
        ) 
        
        pos_embed = pos_embed.permute(2, 0, 1)  

        if self.pos_embed_learned is not None and features_flat.shape[0] == self.pos_embed_learned.shape[1]:
            learned_pos = self.pos_embed_learned.permute(1, 0, 2)  
            learned_pos = learned_pos.expand(-1, B, -1)  
            pos_embed = pos_embed + learned_pos
        
        return features_flat, xyz, pos_embed