"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


class PreprocessedVolumeDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        patch_size: int = 128,
        patches_per_volume: int = 4,
        use_labeled: bool = False
    ):
        if use_labeled:
            pattern = '*_labeled.npz'
        else:
            pattern = '*.npz'
        
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        
        self.total_patches = len(self.file_paths) * patches_per_volume
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        volume_idx = idx // self.patches_per_volume
        volume_idx = volume_idx % len(self.file_paths)  
        
        data = np.load(self.file_paths[volume_idx])
        volume = data['volume'] 
        patch = self._extract_random_patch(volume)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0) 
        
        return patch_tensor
    
    def _extract_random_patch(self, volume: np.ndarray) -> np.ndarray:
        Z, Y, X = volume.shape
        ps = self.patch_size
        
        max_z = max(0, Z - ps)
        max_y = max(0, Y - ps)
        max_x = max(0, X - ps)
        
        start_z = np.random.randint(0, max_z + 1) if max_z > 0 else 0
        start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        
        patch = volume[
            start_z:start_z + ps,
            start_y:start_y + ps,
            start_x:start_x + ps
        ]
        
        if patch.shape != (ps, ps, ps):
            pad_z = ps - patch.shape[0]
            pad_y = ps - patch.shape[1]
            pad_x = ps - patch.shape[2]
            
            patch = np.pad(
                patch,
                [(0, pad_z), (0, pad_y), (0, pad_x)],
                mode='constant',
                constant_values=0
            )
        
        return patch


def create_patch_mask(volume: torch.Tensor, mask_ratio: float = 0.75, patch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    B, C, D, H, W = volume.shape
    
    num_patches_d = D // patch_size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_d * num_patches_h * num_patches_w
    
    num_masked = int(total_patches * mask_ratio)
    
    patch_indices = torch.randperm(total_patches)[:num_masked]
    
    masked_volume = volume.clone()
    
    for patch_idx in patch_indices:
        pd = patch_idx // (num_patches_h * num_patches_w)
        remainder = patch_idx % (num_patches_h * num_patches_w)
        ph = remainder // num_patches_w
        pw = remainder % num_patches_w
        
        d_start = pd * patch_size
        h_start = ph * patch_size
        w_start = pw * patch_size
        
        masked_volume[:, :,
                      d_start:d_start + patch_size,
                      h_start:h_start + patch_size,
                      w_start:w_start + patch_size] = -1.0
    
    return masked_volume



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        diff_z = skip_connection.size(2) - x.size(2)
        diff_y = skip_connection.size(3) - x.size(3)
        diff_x = skip_connection.size(4) - x.size(4)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2,
                      diff_z // 2, diff_z - diff_z // 2])
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv_block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_filters: int = 16):
        super().__init__()
        
        # Encoder
        self.down1 = ConvBlock(in_channels, n_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = ConvBlock(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = ConvBlock(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down4 = ConvBlock(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16)
        
        # Decoder
        self.up4 = UpConvBlock(n_filters * 16, n_filters * 8)
        self.up3 = UpConvBlock(n_filters * 8, n_filters * 4)
        self.up2 = UpConvBlock(n_filters * 4, n_filters * 2)
        self.up1 = UpConvBlock(n_filters * 2, n_filters)
        
        # Output
        self.out_conv = nn.Conv3d(n_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        c4 = self.down4(p3); p4 = self.pool4(c4)
        b = self.bottleneck(p4)
        
        u4 = self.up4(b, c4)
        u3 = self.up3(u4, c3)
        u2 = self.up2(u3, c2)
        u1 = self.up1(u2, c1)
        
        out = self.out_conv(u1)
        return out



def ssl_training_pipeline(
    data_dir: str = 'preprocessed_data',
    model_path: str = '3D_unet_ssl_weights.pth',
    num_epochs: int = 50,
    batch_size: int = 8,
    patch_size: int = 128,
    patches_per_volume: int = 4,
    use_labeled_only: bool = False
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset with patch extraction
    ssl_dataset = PreprocessedVolumeDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume,
        use_labeled=use_labeled_only
    )
    
    ssl_dataloader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Batch size: {batch_size}")
    print(f"Patch size: {patch_size}³")
    print(f"Batches per epoch: {len(ssl_dataloader)}")
    print("=" * 60)
    
    # Model
    ssl_model = UNet3D(in_channels=1, out_channels=1, n_filters=16).to(device)
    total_params = sum(p.numel() for p in ssl_model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(ssl_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    ssl_model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        for batch_idx, original_patches in enumerate(ssl_dataloader):
            original_patches = original_patches.to(device)
            
            masked_patches = create_patch_mask(
                original_patches,
                mask_ratio=0.75,
                patch_size=8  
            )
            
            optimizer.zero_grad()
            
            reconstructed = ssl_model(masked_patches)
            
            loss = criterion(reconstructed, original_patches)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(ssl_dataloader)} | Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(ssl_dataloader)
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Avg Loss: {avg_loss:.6f}")
      
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{model_path}.epoch_{epoch+1}.pth"
            torch.save(ssl_model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}\n")
    
    torch.save(ssl_model.state_dict(), model_path)
    print(f"Training Complete! Model saved: {model_path}")


if __name__ == '__main__':
    ssl_training_pipeline(
        data_dir='preprocessed_data',
        model_path='path to your pretrained 3D UNet encoder',
        num_epochs=50,
        batch_size=8,       
        patch_size=128,   
        patches_per_volume=4,  
        use_labeled_only=False 
    )
