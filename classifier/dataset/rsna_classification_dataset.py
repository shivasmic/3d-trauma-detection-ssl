"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RSNAClassificationDataset(Dataset):
    
    
    def __init__(self, data_dir, csv_path, train_images_dir, split='train', 
                 train_ratio=0.70, val_ratio=0.15, seed=42, augment=True):
        
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.train_images_dir = train_images_dir
        self.split = split
        self.augment = augment and (split == 'train')
        
        
        print("1. Loading patient labels from train_2024.csv...")
        self.df_patients = pd.read_csv(csv_path)
        
        self.patient_to_series = {}
        
        for patient_dir in glob.glob(os.path.join(train_images_dir, "*")):
            patient_id = int(os.path.basename(patient_dir))
            series_dirs = glob.glob(os.path.join(patient_dir, "*"))
            series_ids = [int(os.path.basename(s)) for s in series_dirs]
            self.patient_to_series[patient_id] = series_ids
        
        total_series_in_dirs = sum(len(s) for s in self.patient_to_series.values())
        
        available_volumes = set()
        
        for vol_dir in [data_dir, data_dir.replace('preprocessed_data', 'preprocessed_ssl')]:
            if os.path.exists(vol_dir):
                files = os.listdir(vol_dir)
                for f in files:
                    if f.endswith('.npz'):
                        series_id = int(f.split('_')[0])
                        available_volumes.add(series_id)
        
        print(f"   Volume files found: {len(available_volumes)}")
        
        self.series_data = []
        
        for patient_id, series_list in self.patient_to_series.items():
            patient_rows = self.df_patients[self.df_patients['patient_id'] == patient_id]
            
            if len(patient_rows) > 0:
                patient_labels = patient_rows.iloc[0]
                
                for series_id in series_list:
                    if series_id in available_volumes:
                        self.series_data.append({
                            'series_id': series_id,
                            'patient_id': patient_id,
                            'labels': patient_labels
                        })
        
        print(f"   Matched series (have both volume + labels): {len(self.series_data)}")
        
        np.random.seed(seed)
        n_total = len(self.series_data)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == 'train':
            self.indices = indices[:n_train]
        elif split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.series_data = [self.series_data[i] for i in self.indices]
        
        print(f"   {split} split: {len(self.series_data)} samples")
        
        self.label_columns = [
            'bowel_healthy', 'bowel_injury',
            'liver_healthy', 'liver_high',
            'kidney_high', 'spleen_healthy',
            'extravasation_injury'
        ]
        
        if split == 'train':
            self.calculate_class_weights()
        
        print(f"{'='*60}\n")
    
    def calculate_class_weights(self):
        print("Class distribution (training set):")
        print("-" * 60)
        
        self.pos_weights = []
        
        for col in self.label_columns:
            pos_count = sum(1 for item in self.series_data if item['labels'][col] == 1)
            neg_count = len(self.series_data) - pos_count
            pos_weight = neg_count / (pos_count + 1e-8)
            self.pos_weights.append(pos_weight)
            
            print(f"{col:25s} | Pos: {pos_count:4d} | Neg: {neg_count:4d} | Weight: {pos_weight:.2f}")
        
        print("-" * 60 + "\n")
    
    def __len__(self):
        return len(self.series_data)
    
    def __getitem__(self, idx):
        item = self.series_data[idx]
        series_id = item['series_id']
        patient_labels = item['labels']
        
        npz_paths = [
            os.path.join(self.data_dir, f'{series_id}_labeled.npz'),
            os.path.join(self.data_dir, f'{series_id}_unlabeled.npz'),
            os.path.join(self.data_dir.replace('preprocessed_data', 'preprocessed_ssl'), 
                        f'{series_id}_labeled.npz'),
            os.path.join(self.data_dir.replace('preprocessed_data', 'preprocessed_ssl'), 
                        f'{series_id}_unlabeled.npz'),
        ]
        
        npz_path = None
        for path in npz_paths:
            if os.path.exists(path):
                npz_path = path
                break
        
        if npz_path is None:
            raise FileNotFoundError(f"No .npz file found for series {series_id}")
        
        # Load volume
        data = np.load(npz_path)
        volume = data['volume'].astype(np.float32)  
        
        if self.augment:
            noise = np.random.randn(*volume.shape).astype(np.float32) * 0.05
            volume = volume + noise
            intensity_shift = np.random.uniform(-0.15, 0.15)
            volume = volume + intensity_shift
            intensity_scale = np.random.uniform(0.85, 1.15)
            volume = volume * intensity_scale
            gamma = np.random.uniform(0.8, 1.2)
            volume = np.sign(volume) * np.power(np.abs(volume), gamma)
            volume = np.clip(volume, 0, 1)
        
        # Get labels (extract 7 classification labels)
        labels = np.array([
            patient_labels[col] for col in self.label_columns
        ], dtype=np.float32)
        
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  
        labels_tensor = torch.from_numpy(labels)  
        
        return {
            'volume': volume_tensor,
            'labels': labels_tensor,
            'series_id': series_id
        }


def collate_fn_classification(batch):
    volumes = torch.stack([item['volume'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    series_ids = [item['series_id'] for item in batch]
    
    return {
        'volumes': volumes,    
        'labels': labels,     
        'series_ids': series_ids
    }

