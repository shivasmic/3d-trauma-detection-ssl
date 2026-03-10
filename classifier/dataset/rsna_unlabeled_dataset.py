"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from dataset.rsna_dataset import RSNATraumaDataset
except ModuleNotFoundError:
    from rsna_dataset import RSNATraumaDataset


class RSNAUnlabeledDataset(Dataset):
    def __init__(self, data_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

        if len(self.files) == 0:
            raise ValueError(
                f"No .npz files found in {data_dir}. Make sure your preprocessed SSL volumes are saved as .npz files"
            )

        print(f"RSNAUnlabeledDataset: {len(self.files)} unlabeled volumes loaded from {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx])
        volume = data['volume'].astype(np.float32)          
        volume = volume[np.newaxis, ...]                     
        volume = torch.from_numpy(volume)                    

        weak_volume   = RSNATraumaDataset._weak_augment(volume)
        strong_volume = RSNATraumaDataset._strong_augment(volume)
        series_id = os.path.splitext(os.path.basename(self.files[idx]))[0]
        if series_id.endswith('_unlabeled'):
            series_id = series_id[: -len('_unlabeled')]

        min_coords = torch.zeros(3)
        max_coords = torch.tensor([1024.0, 336.0, 336.0])

        return {
            'weak_volume':   weak_volume,    
            'strong_volume': strong_volume,  
            'volume_dims':   (min_coords, max_coords),
            'series_id':     series_id,
        }


def collate_fn_unlabeled(batch: list) -> dict:
    weak_volumes   = torch.stack([item['weak_volume']   for item in batch])   
    strong_volumes = torch.stack([item['strong_volume'] for item in batch])   

    min_coords = torch.stack([item['volume_dims'][0] for item in batch])      
    max_coords = torch.stack([item['volume_dims'][1] for item in batch])    

    series_ids = [item['series_id'] for item in batch]

    return {
        'weak_volume':   weak_volumes,
        'strong_volume': strong_volumes,
        'volume_dims':   (min_coords, max_coords),
        'series_ids':    series_ids,
    }
