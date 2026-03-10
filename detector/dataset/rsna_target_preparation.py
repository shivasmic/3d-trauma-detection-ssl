"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import numpy as np
from dataset.coordinate_adapter import RSNACoordinateAdapter
from dataset.coordinate_adapter import get_rsna_coordinate_adapter


def prepare_targets_rsna(batch, dataset_config, device):
   
    batch_size = batch['dicom_volumes'].shape[0]
    coord_adapter = get_rsna_coordinate_adapter()
    
    min_coords, max_coords = batch['volume_dims']
    min_coords = min_coords.to(device)
    max_coords = max_coords.to(device)
    volume_range = max_coords - min_coords  
    
    targets = {}
    

    max_boxes = 5
    
    gt_box_corners = torch.zeros(batch_size, max_boxes, 8, 3, device=device)
    gt_box_centers = torch.zeros(batch_size, max_boxes, 3, device=device)
    gt_box_centers_normalized = torch.zeros(batch_size, max_boxes, 3, device=device)
    gt_box_sizes = torch.zeros(batch_size, max_boxes, 3, device=device)
    gt_box_sizes_normalized = torch.zeros(batch_size, max_boxes, 3, device=device)
    gt_box_sem_cls_label = torch.zeros(batch_size, max_boxes, dtype=torch.long, device=device)
    gt_box_present = torch.zeros(batch_size, max_boxes, dtype=torch.float32, device=device)
    
    gt_box_angles = torch.zeros(batch_size, max_boxes, device=device)
    gt_angle_class_label = torch.zeros(batch_size, max_boxes, dtype=torch.long, device=device)
    gt_angle_residual_label = torch.zeros(batch_size, max_boxes, device=device)
    
    nactual_gt = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        if batch['has_labels'][b]:
            nactual_gt[b] = 1
            gt_box_present[b, 0] = 1.0
            
            center_voxel = batch['bbox_center_voxel'][b:b+1].to(device)  
            size_voxel = batch['bbox_size_voxel'][b:b+1].to(device)  
            
            center_physical = coord_adapter.voxel_to_physical(center_voxel)  
            spacing = torch.tensor(coord_adapter.voxel_spacing, device=device, dtype=torch.float32)
            size_physical = size_voxel * spacing  
            
            gt_box_centers[b, 0] = center_physical[0]
            gt_box_sizes[b, 0] = size_physical[0]
            
            center_normalized = (center_physical[0] - min_coords[b]) / volume_range[b]
            size_normalized = size_physical[0] / volume_range[b]
            
            gt_box_centers_normalized[b, 0] = center_normalized
            gt_box_sizes_normalized[b, 0] = size_normalized
            
            gt_box_corners[b, 0] = batch['bbox_corners_physical'][b].to(device)  # [8, 3]
            
            gt_box_sem_cls_label[b, 0] = 0
        else:
            nactual_gt[b] = 0
    
    targets = {
        'gt_box_corners': gt_box_corners, 
        'gt_box_centers': gt_box_centers, 
        'gt_box_centers_normalized': gt_box_centers_normalized,  
        'gt_box_sizes': gt_box_sizes,  
        'gt_box_sizes_normalized': gt_box_sizes_normalized, 
        'gt_box_sem_cls_label': gt_box_sem_cls_label,  
        'gt_box_present': gt_box_present,  
        'gt_box_angles': gt_box_angles,  
        'gt_angle_class_label': gt_angle_class_label,  
        'gt_angle_residual_label': gt_angle_residual_label,  
        'nactual_gt': nactual_gt,  
        'scan_idx': torch.arange(batch_size, device=device),  
    }
    
    return targets


class RSNADatasetConfig:
    def __init__(self):
        self.num_semcls = 1
        
        self.num_angle_bin = 1 
        
        self.mean_size_arr = np.array([
            [30.0, 40.0, 40.0],  
        ])
        
        self.class_names = ['injury']
        
        self.volume_shape = (512, 336, 336)
        self.voxel_spacing = (2.0, 1.0, 1.0)
        self.physical_dims = (1024.0, 336.0, 336.0)
    
    def box_parametrization_to_corners(self, center, size, angle=None):
        device = center.device
        dtype = center.dtype
        
        half_size = size / 2
        
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
        
        corners = center.unsqueeze(2) + half_size.unsqueeze(2) * offsets.unsqueeze(0).unsqueeze(0)
        
        return corners 


def test_target_preparation():
    
    B = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    coord_adapter = RSNACoordinateAdapter()
    
    bbox_center_voxel = torch.tensor([
        [256.0, 168.0, 168.0],  
        [0.0, 0.0, 0.0],         
    ], device=device)
    
    bbox_size_voxel = torch.tensor([
        [20.0, 30.0, 30.0],      
        [0.0, 0.0, 0.0],         
    ], device=device)
    
    bbox_corners = coord_adapter.bbox_voxel_to_corners_physical(
        bbox_center_voxel.unsqueeze(1), 
        bbox_size_voxel.unsqueeze(1)
    ).squeeze(1)  
    
    min_coords = torch.zeros(B, 3, device=device)
    max_coords = torch.tensor(coord_adapter.physical_dims, device=device).unsqueeze(0).expand(B, -1)
    
    batch = {
        'dicom_volumes': torch.randn(B, 1, 512, 336, 336, device=device),
        'bbox_center_voxel': bbox_center_voxel,
        'bbox_size_voxel': bbox_size_voxel,
        'bbox_corners_physical': bbox_corners,
        'volume_dims': (min_coords, max_coords),
        'has_labels': torch.tensor([True, False], device=device),
    }
    
    dataset_config = RSNADatasetConfig()
    
    targets = prepare_targets_rsna(batch, dataset_config, device)
    
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_target_preparation()
