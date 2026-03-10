"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    def __init__(self,
                 center_weight=1.0,
                 size_weight=1.0,
                 cls_weight=1.0,
                 temperature=1.0,
                 start_epoch=0,          
                 warmup_epochs=10,        
                 max_weight=1.0):         

        super().__init__()
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight

    def forward(self, weak_outputs, strong_outputs, epoch=0): 
        if epoch < self.start_epoch:
            device = list(weak_outputs['outputs'].values())[0].device
            return torch.tensor(0.0, device=device), {
                'loss_consistency': 0.0,
                'loss_consistency_center': 0.0,
                'loss_consistency_size': 0.0,
                'loss_consistency_cls': 0.0,
                'ssl_weight': 0.0,
            }
        
        warmup_progress = min(1.0, (epoch - self.start_epoch) / self.warmup_epochs)
        ssl_weight = self.max_weight * warmup_progress
        weak_pred = weak_outputs['outputs']
        strong_pred = strong_outputs['outputs']
        weak_center = weak_pred['center_unnormalized']
        strong_center = strong_pred['center_unnormalized']
        loss_center = F.mse_loss(strong_center, weak_center.detach())
        weak_size = weak_pred['size_unnormalized']
        strong_size = strong_pred['size_unnormalized']
        loss_size = F.mse_loss(strong_size, weak_size.detach())


        if 'sem_cls_logits' in weak_pred:
            weak_logits = weak_pred['sem_cls_logits']       
            strong_logits = strong_pred['sem_cls_logits']
        else:
            weak_logits = weak_pred['sem_cls_prob']
            strong_logits = strong_pred['sem_cls_prob']

        weak_soft = F.softmax(weak_logits.detach() / self.temperature, dim=-1)
        strong_log_soft = F.log_softmax(strong_logits / self.temperature, dim=-1)

        loss_cls = F.kl_div(
            strong_log_soft,
            weak_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)  

        total_loss = ssl_weight * (
            self.center_weight * loss_center +
            self.size_weight  * loss_size +
            self.cls_weight   * loss_cls
        )

        loss_detail = {
            'loss_consistency':        total_loss.item(),
            'loss_consistency_center': loss_center.item(),
            'loss_consistency_size':   loss_size.item(),
            'loss_consistency_cls':    loss_cls.item(),
            'ssl_weight':              ssl_weight,  
        }

        return total_loss, loss_detail



def get_consistency_weight(epoch, warmup_epochs=10, max_weight=1.0, start_epoch=0):
    if epoch < start_epoch:
        return 0.0
    
    progress = min(1.0, (epoch - start_epoch) / warmup_epochs)
    return max_weight * progress
