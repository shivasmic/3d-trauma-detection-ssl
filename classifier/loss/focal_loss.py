"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()


def build_loss_function(cfg, pos_weights):
    loss_type = cfg.type if hasattr(cfg, 'type') else 'bce_with_logits'
    
    if loss_type == 'focal_loss':
        print(f"Using Focal Loss (alpha={cfg.focal_alpha}, gamma={cfg.focal_gamma})")
        return FocalLoss(
            alpha=cfg.focal_alpha,
            gamma=cfg.focal_gamma,
            pos_weight=pos_weights
        )
    else:
        print("Using BCE with Logits Loss")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)