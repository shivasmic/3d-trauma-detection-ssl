"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn


class RSNAClassificationModel(nn.Module):
    def __init__(self, unet_encoder, num_classes=7, dropout=0.5):
        super().__init__()
        
        self.encoder = unet_encoder
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),      
            nn.Flatten(),                  
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),           
            nn.Linear(128, num_classes)    
        )
        
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        c1 = self.encoder.down1(x)
        p1 = self.encoder.pool1(c1)
        c2 = self.encoder.down2(p1)
        p2 = self.encoder.pool2(c2)
        c3 = self.encoder.down3(p2)
        p3 = self.encoder.pool3(c3)
        c4 = self.encoder.down4(p3)
        p4 = self.encoder.pool4(c4)
        
        features = self.encoder.bottleneck(p4)  
        
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        
        return probs, preds


def build_classification_model(unet_encoder, num_classes=7, dropout=0.5):
    model = RSNAClassificationModel(unet_encoder, num_classes, dropout)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = encoder_params + classifier_params
    
    print("CLASSIFICATION MODEL")
    print(f"Encoder parameters:    {encoder_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Dropout rate:          {dropout}")    
    return model

