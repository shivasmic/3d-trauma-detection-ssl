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
    from coordinate_adapter import get_rsna_coordinate_adapter



class RSNATraumaDataset(Dataset):
    
    def __init__(self, data_dir, split='train', 
                 transform=None, train_ratio=0.70, val_ratio=0.15, seed=42):
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.coord_adapter = get_rsna_coordinate_adapter()
        
        all_files = sorted(glob.glob(os.path.join(data_dir, '*_liver.npz')))
        
        if len(all_files) == 0:
            raise ValueError(f"No *_liver.npz files found in {data_dir}")
        
        if split is None:
            self.files = all_files
            print(f"RSNA Liver dataset (no split): {len(self.files)} samples")
        else:
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
                raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', 'test', or None")
            
            self.files = [all_files[i] for i in selected_indices]
            
            print(f"RSNA Liver {split} dataset: {len(self.files)} samples")
            if split == 'train':
                print(f"  Total liver files: {len(all_files)}")
                print(f"  Train: {num_train} ({train_ratio*100:.0f}%)")
                print(f"  Val:   {num_val} ({val_ratio*100:.0f}%)")
                print(f"  Test:  {num_test} ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        # Load one sample to check format
        if len(self.files) > 0:
            sample = np.load(self.files[0])
            if split == 'train' or split is None:  
                print(f"Sample keys: {list(sample.keys())}")
                print(f"Volume shape: {sample['volume'].shape}")
                print(f"BBox center: {sample['bbox_center']}")
                print(f"BBox size: {sample['bbox_size']}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        volume = data['volume'].astype(np.float32)
        volume = volume[np.newaxis, ...]  # Add channel dimension
        
        series_id = os.path.basename(self.files[idx]).replace('_liver.npz', '')
        
        bbox_center_voxel = data['bbox_center'].astype(np.float32)  # [3]
        bbox_size_voxel = data['bbox_size'].astype(np.float32)  # [3]
        mask = data['mask'].astype(np.uint8)  # [512, 336, 336]
        
        bbox_center_voxel = bbox_center_voxel[np.newaxis, np.newaxis, ...]
        bbox_size_voxel = bbox_size_voxel[np.newaxis, np.newaxis, ...]
        
        bbox_center_voxel = torch.from_numpy(bbox_center_voxel)
        bbox_size_voxel = torch.from_numpy(bbox_size_voxel)
        
        bbox_corners_physical = self.coord_adapter.bbox_voxel_to_corners_physical(
            bbox_center_voxel, bbox_size_voxel
        ) 
        
        bbox_corners_physical = bbox_corners_physical.squeeze(1)
        
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
            'has_label': True,  
        }


def collate_fn(batch):
    volumes = torch.stack([item['dicom_volumes'] for item in batch])  # [B, 1, 512, 336, 336]
    
    bbox_centers = torch.cat([item['bbox_center_voxel'] for item in batch], dim=0)  # [B, 3]
    bbox_sizes = torch.cat([item['bbox_size_voxel'] for item in batch], dim=0)  # [B, 3]
    bbox_corners = torch.cat([item['bbox_corners_physical'] for item in batch], dim=0)  # [B, 8, 3]
    
    masks = torch.stack([item['mask'] for item in batch])  # [B, 512, 336, 336]
    
    min_coords = torch.stack([item['volume_dims'][0] for item in batch])  # [B, 3]
    max_coords = torch.stack([item['volume_dims'][1] for item in batch])  # [B, 3]
    
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
    train_dataset = RSNATraumaDataset(data_dir, split='train')
    val_dataset = RSNATraumaDataset(data_dir, split='val')
    test_dataset = RSNATraumaDataset(data_dir, split='test')
    
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


# Test script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    train_dataset = RSNATraumaDataset(data_dir, split='train')
    val_dataset = RSNATraumaDataset(data_dir, split='val')
    test_dataset = RSNATraumaDataset(data_dir, split='test')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("No liver data found!")
        sys.exit(1)
    
    # Load one sample
    sample = train_dataset[0]
    
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    print(f"  BBox center (voxels Z,Y,X): {sample['bbox_center_voxel'][0].numpy()}")
    print(f"  BBox size (voxels Z,Y,X): {sample['bbox_size_voxel'][0].numpy()}")
    print(f"  BBox corner 0 (mm Z,Y,X): {sample['bbox_corners_physical'][0, 0].numpy()}")
    print(f"  BBox corner 6 (mm Z,Y,X): {sample['bbox_corners_physical'][0, 6].numpy()}")
    print(f"  Volume dims: {sample['volume_dims'][0].numpy()} to {sample['volume_dims'][1].numpy()} mm")
    
    # Create dataloaders
    print("Testing Dataloaders (batch_size=2)")
    
    train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=2, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print("\nBatch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, tuple) and isinstance(value[0], torch.Tensor):
            print(f"  {key}: ({value[0].shape}, {value[1].shape})")
        else:
            print(f"  {key}: {type(value)}")
    
    print("\n Liver dataset loader working correctly!")
