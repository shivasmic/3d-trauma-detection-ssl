"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl

"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

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
        
        # Decoder (kept for weight compatibility)
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
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)      
        p1 = self.pool1(c1)     
        
        c2 = self.down2(p1)     
        p2 = self.pool2(c2)     
        
        c3 = self.down3(p2)     
        p3 = self.pool3(c3)     
        
        c4 = self.down4(p3)     
        p4 = self.pool4(c4)     
        
        bottleneck = self.bottleneck(p4)  
        
        return bottleneck



def load_pretrained_unet_encoder(
    pretrained_path: str = 'your path to pretrained 3D Unet Encoder Weights',
    freeze_encoder: bool = True,
    device: str = 'cuda'
) -> UNet3D:
    
    
    model = UNet3D(in_channels=1, out_channels=1, n_filters=16)
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    if list(checkpoint.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  
            new_checkpoint[name] = v
        checkpoint = new_checkpoint
    
    model.load_state_dict(checkpoint)
    
    if freeze_encoder:
        frozen_params = 0
        for name, param in model.named_parameters():
            if any(x in name for x in ['down', 'pool', 'bottleneck']):
                param.requires_grad = False
                frozen_params += 1
        print(f"Decoder remains trainable for fine-tuning")
    else:
        print(f"✓ All parameters trainable")
    
    model = model.to(device)
    print(f"✓ Model moved to {device}")
    return model
