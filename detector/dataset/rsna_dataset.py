"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from dataset.coordinate_adapter import get_rsna_coordinate_adapter
except ModuleNotFoundError:
    from dataset.coordinate_adapter import get_rsna_coordinate_adapter



class RSNATraumaDataset(Dataset):
    def __init__(self, data_dir, split='train', use_labeled_only=True, 
                 transform=None, train_ratio=0.70, val_ratio=0.15, seed=42):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.coord_adapter = get_rsna_coordinate_adapter()
        
        if use_labeled_only:
            all_files = sorted(glob.glob(os.path.join(data_dir, '*_labeled.npz')))
        else:
            all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        if len(all_files) == 0:
            raise ValueError(f"No files found in {data_dir}")
        
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))
        
        num_train = int(train_ratio * len(all_files))
        num_val = int(val_ratio * len(all_files))
        num_test = len(all_files) - num_train - num_val
        
        if split == 'train':
            selected_indices = indices[:num_train]
        elif split == 'val':
            selected_indices = indices[num_train:num_train + num_val]
        elif split == 'test':
            selected_indices = indices[num_train + num_val:]
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
        
        self.files = [all_files[i] for i in selected_indices]
        
        print(f"RSNA {split} dataset: {len(self.files)} samples")
        if split == 'train':
            print(f"  Total files: {len(all_files)}")
            print(f"  Train: {num_train} ({train_ratio*100:.0f}%)")
            print(f"  Val:   {num_val} ({val_ratio*100:.0f}%)")
            print(f"  Test:  {num_test} ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        if len(self.files) > 0:
            sample = np.load(self.files[0])
            if split == 'train':  # Only print once
                print(f"Sample keys: {list(sample.keys())}")
                print(f"Volume shape: {sample['volume'].shape}")
                if 'bbox_center' in sample:
                    print(f"BBox center: {sample['bbox_center']}")
                    print(f"BBox size: {sample['bbox_size']}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        volume = data['volume'].astype(np.float32)
        volume = volume[np.newaxis, ...]  
        
        series_id = os.path.basename(self.files[idx]).replace('_labeled.npz', '').replace('_unlabeled.npz', '')
        
        has_label = 'bbox_center' in data
        
        if has_label:
            bbox_center_voxel = data['bbox_center'].astype(np.float32)  
            bbox_size_voxel = data['bbox_size'].astype(np.float32)  
            mask = data['mask'].astype(np.uint8)  
            
            bbox_center_voxel = bbox_center_voxel[np.newaxis, np.newaxis, ...]
            bbox_size_voxel = bbox_size_voxel[np.newaxis, np.newaxis, ...]
            
            bbox_center_voxel = torch.from_numpy(bbox_center_voxel)
            bbox_size_voxel = torch.from_numpy(bbox_size_voxel)
            
            bbox_corners_physical = self.coord_adapter.bbox_voxel_to_corners_physical(
                bbox_center_voxel, bbox_size_voxel
            ) 
            
            bbox_corners_physical = bbox_corners_physical.squeeze(1)
        else:
            bbox_center_voxel = torch.zeros(1, 3)
            bbox_size_voxel = torch.zeros(1, 3)
            bbox_corners_physical = torch.zeros(1, 8, 3)
            mask = np.zeros((512, 336, 336), dtype=np.uint8)
        
        volume = torch.from_numpy(volume)
        mask = torch.from_numpy(mask)
        
        min_coords = torch.zeros(3)  
        max_coords = torch.tensor(self.coord_adapter.physical_dims)  
        
        return {
            'dicom_volumes': volume,  
            'bbox_center_voxel': bbox_center_voxel, 
            'bbox_size_voxel': bbox_size_voxel,
            'bbox_corners_physical': bbox_corners_physical,  
            'mask': mask,  
            'volume_dims': (min_coords, max_coords),
            'series_id': series_id,
            'has_label': has_label,
        }

   
    @staticmethod
    def _weak_augment(volume: torch.Tensor) -> torch.Tensor:
        v = volume.clone()

        shift = torch.empty(1).uniform_(-0.05, 0.05).item()
        v = v + shift

        scale = torch.empty(1).uniform_(0.90, 1.10).item()
        v = v * scale

        noise_std = torch.empty(1).uniform_(0.0, 0.02).item()
        if noise_std > 0:
            v = v + torch.randn_like(v) * noise_std

        v = v.clamp(0.0, 1.0)
        return v

    @staticmethod
    def _strong_augment(volume: torch.Tensor) -> torch.Tensor:
        v = volume.clone()

        shift = torch.empty(1).uniform_(-0.15, 0.15).item()
        v = v + shift

        scale = torch.empty(1).uniform_(0.70, 1.30).item()
        v = v * scale

        noise_std = torch.empty(1).uniform_(0.0, 0.08).item()
        if noise_std > 0:
            v = v + torch.randn_like(v) * noise_std

        v = v.clamp(1e-6, 1.0 - 1e-6)
        gamma = torch.empty(1).uniform_(0.5, 2.0).item()
        v = v.pow(gamma)

        if torch.rand(1).item() < 0.5:
            v = v.flip(1)   
        if torch.rand(1).item() < 0.5:
            v = v.flip(2)   
        if torch.rand(1).item() < 0.5:
            v = v.flip(3)  

        num_patches = torch.randint(1, 4, (1,)).item()   
        _, D, H, W = v.shape
        for _ in range(num_patches):
            side = torch.randint(16, 65, (1,)).item()   
            z0 = torch.randint(0, max(1, D - side), (1,)).item()
            y0 = torch.randint(0, max(1, H - side), (1,)).item()
            x0 = torch.randint(0, max(1, W - side), (1,)).item()
            v[:, z0:z0+side, y0:y0+side, x0:x0+side] = 0.0

        v = v.clamp(0.0, 1.0)
        return v


def collate_fn(batch):
    volumes = torch.stack([item['dicom_volumes'] for item in batch])
    

    bbox_centers = torch.cat([item['bbox_center_voxel'] for item in batch], dim=0) 
    bbox_sizes = torch.cat([item['bbox_size_voxel'] for item in batch], dim=0)  
    bbox_corners = torch.cat([item['bbox_corners_physical'] for item in batch], dim=0)  
    
    masks = torch.stack([item['mask'] for item in batch])  
    
    min_coords = torch.stack([item['volume_dims'][0] for item in batch])  
    max_coords = torch.stack([item['volume_dims'][1] for item in batch])  
    
    series_ids = [item['series_id'] for item in batch]
    has_labels = torch.tensor([item['has_label'] for item in batch])
    
    return {
        'dicom_volumes': volumes,  
        'bbox_center_voxel': bbox_centers,  
        'bbox_size_voxel': bbox_sizes,  
        'bbox_corners_physical': bbox_corners,  
        'masks': masks,  
        'volume_dims': (min_coords, max_coords),
        'series_ids': series_ids,
        'has_labels': has_labels,
    }


def create_dataloaders(data_dir, batch_size=2, num_workers=4):
    train_dataset = RSNATraumaDataset(data_dir, split='train', use_labeled_only=True)
    val_dataset = RSNATraumaDataset(data_dir, split='val', use_labeled_only=True)
    test_dataset = RSNATraumaDataset(data_dir, split='test', use_labeled_only=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


