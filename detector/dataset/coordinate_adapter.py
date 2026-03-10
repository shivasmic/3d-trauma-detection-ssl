"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import numpy as np

class RSNACoordinateAdapter:
    
    def __init__(self):
        self.voxel_spacing = np.array([2.0, 1.0, 1.0], dtype=np.float32)  # (Z, Y, X) in mm
        self.volume_shape = np.array([512, 336, 336], dtype=np.int32)     # (Z, Y, X) in voxels

        self.physical_dims = self.volume_shape * self.voxel_spacing
        
        self.origin = np.zeros(3, dtype=np.float32)
        
        
    def voxel_to_physical(self, voxel_coords):
        if isinstance(voxel_coords, np.ndarray):
            return voxel_coords * self.voxel_spacing + self.origin
        else:
            spacing = torch.tensor(self.voxel_spacing, device=voxel_coords.device, dtype=voxel_coords.dtype)
            origin = torch.tensor(self.origin, device=voxel_coords.device, dtype=voxel_coords.dtype)
            return voxel_coords * spacing + origin
    
    def physical_to_voxel(self, physical_coords):
        if isinstance(physical_coords, np.ndarray):
            return (physical_coords - self.origin) / self.voxel_spacing
        else:
            spacing = torch.tensor(self.voxel_spacing, device=physical_coords.device, dtype=physical_coords.dtype)
            origin = torch.tensor(self.origin, device=physical_coords.device, dtype=physical_coords.dtype)
            return (physical_coords - origin) / spacing
    
    def normalize_coordinates(self, physical_coords):
        if isinstance(physical_coords, np.ndarray):
            return physical_coords / self.physical_dims
        else:
            dims = torch.tensor(self.physical_dims, device=physical_coords.device, dtype=physical_coords.dtype)
            return physical_coords / dims
    
    def denormalize_coordinates(self, normalized_coords):
        if isinstance(normalized_coords, np.ndarray):
            return normalized_coords * self.physical_dims
        else:
            dims = torch.tensor(self.physical_dims, device=normalized_coords.device, dtype=normalized_coords.dtype)
            return normalized_coords * dims
    
    def bbox_voxel_to_corners_physical(self, bbox_center_voxel, bbox_size_voxel):
        device = bbox_center_voxel.device
        dtype = bbox_center_voxel.dtype
        
        center_phys = self.voxel_to_physical(bbox_center_voxel)
        
        # Convert size to physical (accounting for anisotropic spacing)
        spacing = torch.tensor(self.voxel_spacing, device=device, dtype=dtype)
        size_phys = bbox_size_voxel * spacing
        
        half_size = size_phys / 2
        
        
        offsets = torch.tensor([
            [-1, -1, -1],  
            [-1, -1,  1], 
            [-1,  1,  1],  
            [-1,  1, -1], 
            [ 1, -1, -1],  
            [ 1, -1,  1],  
            [ 1,  1,  1],  
            [ 1,  1, -1],  
        ], device=device, dtype=dtype)
        
        
        center_expanded = center_phys.unsqueeze(2)           
        half_size_expanded = half_size.unsqueeze(2)         
        offsets_expanded = offsets.unsqueeze(0).unsqueeze(0) 
        corners = center_expanded + half_size_expanded * offsets_expanded  
        
        return corners  
    
    def get_volume_dims_tensor(self, batch_size, device):
        min_coords = torch.tensor(self.origin, device=device).unsqueeze(0).expand(batch_size, -1)
        max_coords = torch.tensor(self.origin + self.physical_dims, device=device).unsqueeze(0).expand(batch_size, -1)
        
        return min_coords, max_coords
    
    def validate_bbox(self, bbox_center_voxel, bbox_size_voxel):
        half_size = bbox_size_voxel / 2
        
        min_coords = bbox_center_voxel - half_size
        max_coords = bbox_center_voxel + half_size
        
        volume_shape = torch.tensor(self.volume_shape, device=bbox_center_voxel.device, dtype=bbox_center_voxel.dtype)
        
        valid = (
            (min_coords >= 0).all(dim=-1) & 
            (max_coords <= volume_shape).all(dim=-1)
        )
        
        return valid


_adapter = None

def get_rsna_coordinate_adapter():
    """Get singleton coordinate adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = RSNACoordinateAdapter()
    return _adapter

