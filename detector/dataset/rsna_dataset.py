"""
Dataset loader for RSNA preprocessed trauma data.

Loads .npz files with format:
- volume: [512, 336, 336] normalized to [0, 1]
- bbox_center: [3] in voxel indices (Z, Y, X)
- bbox_size: [3] in voxels (Z, Y, X)
- mask: [512, 336, 336] binary mask
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
    """
    Dataset for RSNA Liver Injury Detection.
    
    Loads preprocessed *_liver.npz files and converts ground truth to model format.
    
    Split Strategy (for 98 liver volumes):
    - Train:      78 volumes (80%)
    - Validation: 20 volumes (20%)
    - Test:       25 volumes (in separate holdout folder)
    """
    
    def __init__(self, data_dir, split='train', use_labeled_only=True, 
                 transform=None, train_ratio=0.80, val_ratio=0.20, seed=42):
        """
        Args:
            data_dir: Directory containing *_liver.npz files
            split: 'train', 'val', or None (None = use all files)
            use_labeled_only: If True, only load files with '_liver.npz'
            transform: Optional data augmentation (not implemented yet)
            train_ratio: Fraction for training (default 0.80)
            val_ratio: Fraction for validation (default 0.20)
            seed: Random seed for reproducible splits
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.coord_adapter = get_rsna_coordinate_adapter()
        
        # Find all data files
        if use_labeled_only:
            all_files = sorted(glob.glob(os.path.join(data_dir, '*_liver.npz')))
        else:
            all_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        if len(all_files) == 0:
            raise ValueError(f"No files found in {data_dir}")
        
        # TWO-WAY SPLIT (train/val only, test uses holdout folder)
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))
        
        num_train = int(train_ratio * len(all_files))
        num_val = len(all_files) - num_train
        
        if split == 'train':
            selected_indices = indices[:num_train]
        elif split == 'val':
            selected_indices = indices[num_train:]
        elif split is None:
            selected_indices = indices
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or None. Use holdout folder for test.")
        
        self.files = [all_files[i] for i in selected_indices]
        
        print(f"RSNA {split} dataset: {len(self.files)} samples")
        if split == 'train':
            print(f"  Total files: {len(all_files)}")
            print(f"  Train: {num_train} ({train_ratio*100:.0f}%)")
            print(f"  Val:   {num_val} ({val_ratio*100:.0f}%)")
            print(f"  (Test uses separate holdout folder)")
        
        # Load one sample to check format
        if len(self.files) > 0:
            sample = np.load(self.files[0])
            if split == 'train':  # Only print once
                print(f"Sample keys: {list(sample.keys())}")
                print(f"Volume shape: {sample['volume'].shape}")
                if 'bbox_center' in sample:
                    print(f"BBox center: {sample['bbox_center']}")
                    print(f"BBox size: {sample['bbox_size']}")
    

    @staticmethod
    def _weak_augment(volume):
        v = volume + torch.randn_like(volume) * 0.01
        v = v * (1.0 + (torch.rand(1).item() * 0.04 - 0.02))
        return v.clamp(0.0, 1.0)

    @staticmethod
    def _strong_augment(volume):
        v = volume + torch.randn_like(volume) * 0.05
        v = v * (1.0 + (torch.rand(1).item() * 0.20 - 0.10))
        gamma = 0.8 + torch.rand(1).item() * 0.4
        return v.clamp(0.0, 1.0) ** gamma

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load and process a single sample.
        
        Returns:
            dict with:
                - dicom_volumes: [1, 512, 336, 336] volume
                - bbox_center_voxel: [1, 3] ground truth center in voxels
                - bbox_size_voxel: [1, 3] ground truth size in voxels
                - bbox_corners_physical: [1, 8, 3] ground truth corners in mm
                - mask: [512, 336, 336] binary mask
                - volume_dims: (min_coords, max_coords) in mm
                - series_id: string identifier
        """
        # Load .npz file
        data = np.load(self.files[idx])
        
        # Extract volume [512, 336, 336] -> [1, 512, 336, 336]
        volume = data['volume'].astype(np.float32)
        volume = volume[np.newaxis, ...]  # Add channel dimension
        
        # Get series ID
        series_id = os.path.basename(self.files[idx]).replace('_liver.npz', '').replace('_unlabeled.npz', '')
        
        # Check if labeled
        has_label = 'bbox_center' in data
        
        if has_label:
            # Ground truth bounding box (in voxel indices)
            bbox_center_voxel = data['bbox_center'].astype(np.float32)  # [3]
            bbox_size_voxel = data['bbox_size'].astype(np.float32)  # [3]
            mask = data['mask'].astype(np.uint8)  # [512, 336, 336]
            
            # Add batch dimension
            bbox_center_voxel = bbox_center_voxel[np.newaxis, np.newaxis, ...]
            bbox_size_voxel = bbox_size_voxel[np.newaxis, np.newaxis, ...]
            
            # Convert to torch
            bbox_center_voxel = torch.from_numpy(bbox_center_voxel)
            bbox_size_voxel = torch.from_numpy(bbox_size_voxel)
            
            # Generate 8 corners in physical space (mm)
            bbox_corners_physical = self.coord_adapter.bbox_voxel_to_corners_physical(
                bbox_center_voxel, bbox_size_voxel
            )  # [1, 1, 8, 3]
            
            # Remove extra dimension: [1, 1, 8, 3] -> [1, 8, 3]
            bbox_corners_physical = bbox_corners_physical.squeeze(1)
        else:
            # No label - use dummy values
            bbox_center_voxel = torch.zeros(1, 3)
            bbox_size_voxel = torch.zeros(1, 3)
            bbox_corners_physical = torch.zeros(1, 8, 3)
            mask = np.zeros((512, 336, 336), dtype=np.uint8)
        
        # Convert volume to torch
        volume = torch.from_numpy(volume)
        mask = torch.from_numpy(mask)
        
        # Get volume dimensions in physical space
        min_coords = torch.zeros(3)  # Origin at (0, 0, 0)
        max_coords = torch.tensor(self.coord_adapter.physical_dims)  # [1024, 336, 336] mm
        
        return {
            'dicom_volumes': volume,  # [1, 512, 336, 336]
            'bbox_center_voxel': bbox_center_voxel,  # [1, 3]
            'bbox_size_voxel': bbox_size_voxel,  # [1, 3]
            'bbox_corners_physical': bbox_corners_physical,  # [1, 8, 3]
            'mask': mask,  # [512, 336, 336]
            'volume_dims': (min_coords, max_coords),
            'series_id': series_id,
            'has_label': has_label,
        }


def collate_fn(batch):
    """
    Custom collate function to handle batch of samples.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched dictionary
    """
    # Stack volumes
    volumes = torch.stack([item['dicom_volumes'] for item in batch])  # [B, 1, 512, 336, 336]
    
    # Stack ground truth (may have different number of boxes per image)
    # For now, assume 1 box per image (injury detection)
    bbox_centers = torch.cat([item['bbox_center_voxel'] for item in batch], dim=0)  # [B, 3]
    bbox_sizes = torch.cat([item['bbox_size_voxel'] for item in batch], dim=0)  # [B, 3]
    bbox_corners = torch.cat([item['bbox_corners_physical'] for item in batch], dim=0)  # [B, 8, 3]
    
    # Stack masks
    masks = torch.stack([item['mask'] for item in batch])  # [B, 512, 336, 336]
    
    # Volume dims (same for all)
    min_coords = torch.stack([item['volume_dims'][0] for item in batch])  # [B, 3]
    max_coords = torch.stack([item['volume_dims'][1] for item in batch])  # [B, 3]
    
    # Series IDs and labels
    series_ids = [item['series_id'] for item in batch]
    has_labels = torch.tensor([item['has_label'] for item in batch])
    
    return {
        'dicom_volumes': volumes,  # [B, 1, 512, 336, 336]
        'bbox_center_voxel': bbox_centers,  # [B, 3]
        'bbox_size_voxel': bbox_sizes,  # [B, 3]
        'bbox_corners_physical': bbox_corners,  # [B, 8, 3]
        'masks': masks,  # [B, 512, 336, 336]
        'volume_dims': (min_coords, max_coords),
        'series_ids': series_ids,
        'has_labels': has_labels,
    }


def create_dataloaders(data_dir, batch_size=2, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory with preprocessed *_liver.npz files
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = RSNATraumaDataset(data_dir, split='train', use_labeled_only=True)
    val_dataset = RSNATraumaDataset(data_dir, split='val', use_labeled_only=True)
    
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
    
    return train_loader, val_loader


# Test script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rsna_dataset.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print("="*60)
    print("Testing RSNA Dataset Loader")
    print("="*60)
    
    # Create datasets
    train_dataset = RSNATraumaDataset(data_dir, split='train', use_labeled_only=True)
    val_dataset = RSNATraumaDataset(data_dir, split='val', use_labeled_only=True)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset)}")
    print(f"  (Test: use holdout folder with 25 files)")
    
    if len(train_dataset) == 0:
        print("No data found!")
        sys.exit(1)
    
    # Load one sample
    sample = train_dataset[0]
    
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    print("\nGround truth details:")
    print(f"  BBox center (voxels Z,Y,X): {sample['bbox_center_voxel'][0].numpy()}")
    print(f"  BBox size (voxels Z,Y,X): {sample['bbox_size_voxel'][0].numpy()}")
    print(f"  BBox corner 0 (mm Z,Y,X): {sample['bbox_corners_physical'][0, 0].numpy()}")
    print(f"  BBox corner 6 (mm Z,Y,X): {sample['bbox_corners_physical'][0, 6].numpy()}")
    print(f"  Volume dims: {sample['volume_dims'][0].numpy()} to {sample['volume_dims'][1].numpy()} mm")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("Testing Dataloaders (batch_size=2)")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(data_dir, batch_size=2, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    
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
    
    print("\n✓ Dataset loader working correctly!")
