"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from classifier.dataset.rsna_classification_dataset import RSNAClassificationDataset, collate_fn_classification
from classifier.models.classification_model import build_classification_model
from classifier.models.unet_encoder import load_pretrained_unet_encoder


class Config:
    """Configuration loaded from YAML - NO ARGPARSE!"""
    
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        for section, params in config_dict.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    setattr(self, key, value)
        
        # Print loaded config
        self.print_config()
    
    def print_config(self):
        """Print all loaded settings"""
        print("\n" + "="*80)
        print("LOADED CONFIGURATION")
        print("="*80)
        
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        for key, value in sorted(attrs.items()):
            print(f"  {key:30s} = {value}")
        
        print("="*80)
        print("\nCRITICAL SETTINGS:")
        print(f"  Data directory:    {self.data_dir}")
        print(f"  CSV path:          {self.csv_path}")
        print(f"  Max epochs:        {self.max_epoch}")
        print(f"  Batch size:        {self.batchsize_per_gpu}")
        print(f"  Base LR:           {self.base_lr}")
        print(f"  Encoder LR:        {self.encoder_lr}")
        print(f"  Dropout:           {self.dropout if hasattr(self, 'dropout') else 0.3}")
        print(f"  Unfreeze at:       epoch {self.unfreeze_after_epoch}")
        print(f"  Loss type:         {self.type if hasattr(self, 'type') else 'bce_with_logits'}")
        print(f"  Checkpoint dir:    {self.checkpoint_dir}")
        print("="*80 + "\n")


def train_one_epoch(epoch, model, optimizer, criterion, dataloader, cfg, pos_weights):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Encoder LR warmup (if just unfrozen)
    if hasattr(cfg, '_unfreeze_epoch') and epoch >= cfg._unfreeze_epoch:
        warmup_epochs = cfg.unfreeze_warmup_epochs
        target_lr = cfg._encoder_lr_target
        
        if epoch < cfg._unfreeze_epoch + warmup_epochs:
            warmup_progress = (epoch - cfg._unfreeze_epoch) / warmup_epochs
            current_encoder_lr = warmup_progress * target_lr
            
            for param_group in optimizer.param_groups:
                if 'encoder' in str(param_group):
                    param_group['lr'] = current_encoder_lr
        elif epoch == cfg._unfreeze_epoch + warmup_epochs:
            for param_group in optimizer.param_groups:
                if 'encoder' in str(param_group):
                    param_group['lr'] = target_lr
    
    for batch_idx, batch in enumerate(dataloader):
        volumes = batch['volumes'].cuda()
        labels = batch['labels'].cuda()
        
        # Forward
        logits = model(volumes)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if cfg.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_gradient)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % cfg.log_every == 0:
            print(f"Epoch [{epoch}/{cfg.max_epoch}] "
                  f"Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate(epoch, model, criterion, dataloader, cfg, pos_weights):
    """Evaluate on validation/test set"""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_preds = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            volumes = batch['volumes'].cuda()
            labels = batch['labels'].cuda()
            
            logits = model(volumes)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > cfg.threshold).float()
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    label_names = [
        'bowel_healthy', 'bowel_injury', 'liver_healthy', 'liver_high',
        'kidney_high', 'spleen_healthy', 'extravasation_injury'
    ]
    
    print("\n" + "="*80)
    print(f"EVALUATION - Epoch [{epoch}/{cfg.max_epoch}]")
    print("="*80)
    print(f"Loss: {avg_loss:.4f}\n")
    
    all_accs = []
    all_aucs = []
    
    for i, label_name in enumerate(label_names):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        else:
            auc = -1
        
        all_accs.append(acc)
        if auc > 0:
            all_aucs.append(auc)
        
        auc_str = f"{auc:.3f}" if auc > 0 else "N/A"
        print(f"{label_name:25s} | Acc: {acc:.3f} | AUC: {auc_str:>6s}")
    
    mean_acc = np.mean(all_accs)
    mean_auc = np.mean(all_aucs) if all_aucs else -1
    
    print("-"*80)
    print(f"OVERALL | Acc: {mean_acc:.3f} | AUC: {mean_auc:.3f}" if mean_auc > 0 
          else f"OVERALL | Acc: {mean_acc:.3f}")
    print("="*80 + "\n")
    
    return {'loss': avg_loss, 'mean_accuracy': mean_acc, 'mean_auc': mean_auc}


def save_checkpoint(checkpoint_dir, model, optimizer, epoch, metrics):
    """Save checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    save_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(checkpoint, save_path)
    print(f"✓ Saved checkpoint: {save_path}")


def main():
    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    
    # Load config (NO ARGPARSE!)
    cfg = Config(config_path)
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Create datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_dataset = RSNAClassificationDataset(
        data_dir=cfg.data_dir,
        csv_path=cfg.csv_path,
        train_images_dir=cfg.train_images_dir,
        split='train'
    )
    
    val_dataset = RSNAClassificationDataset(
        data_dir=cfg.data_dir,
        csv_path=cfg.csv_path,
        train_images_dir=cfg.train_images_dir,
        split='val'
    )
    
    test_dataset = RSNAClassificationDataset(
        data_dir=cfg.data_dir,
        csv_path=cfg.csv_path,
        train_images_dir=cfg.train_images_dir,
        split='test'
    )
    
    pos_weights = torch.tensor(train_dataset.pos_weights).float().cuda()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.dataset_num_workers,
        collate_fn=collate_fn_classification,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.dataset_num_workers,
        collate_fn=collate_fn_classification,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.dataset_num_workers,
        collate_fn=collate_fn_classification,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    unet_encoder = load_pretrained_unet_encoder(
        pretrained_path=cfg.pretrained_weights,
        freeze_encoder=cfg.freeze_encoder,
        device='cuda:0'
    )
    
    # Build model with configurable dropout
    model = build_classification_model(
        unet_encoder, 
        num_classes=cfg.num_classes,
        dropout=cfg.dropout if hasattr(cfg, 'dropout') else 0.3
    )
    model = model.cuda()
    
    # Build loss function (BCE or Focal)
    print("\n" + "="*60)
    print("LOSS FUNCTION")
    print("="*60)
    
    if hasattr(cfg, 'type') and cfg.type == 'focal_loss':
        from loss.focal_loss import FocalLoss
        criterion = FocalLoss(
            alpha=cfg.focal_alpha if hasattr(cfg, 'focal_alpha') else 0.25,
            gamma=cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0,
            pos_weight=pos_weights
        )
        print(f"Using Focal Loss")
        print(f"  Alpha (balancing):  {cfg.focal_alpha if hasattr(cfg, 'focal_alpha') else 0.25}")
        print(f"  Gamma (focusing):   {cfg.focal_gamma if hasattr(cfg, 'focal_gamma') else 2.0}")
        print(f"  Pos weights:        Applied ({len(pos_weights)} classes)")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        print("Using BCE with Logits Loss")
        print(f"  Pos weights: Applied ({len(pos_weights)} classes)")
    
    print("="*60)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay
    )
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_val_acc = 0.0
    
    for epoch in range(cfg.max_epoch):
        
        # Train
        train_loss = train_one_epoch(epoch, model, optimizer, criterion, 
                                     train_loader, cfg, pos_weights)
        print(f"Epoch [{epoch}/{cfg.max_epoch}] Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % cfg.eval_every_epoch == 0 or epoch == cfg.max_epoch - 1:
            val_metrics = evaluate(epoch, model, criterion, val_loader, cfg, pos_weights)
            
            if val_metrics['mean_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['mean_accuracy']
                save_checkpoint(cfg.checkpoint_dir, model, optimizer, epoch, val_metrics)
                print(f"✓ New best! Val Acc: {best_val_acc:.3f}")
        
        # Unfreeze encoder
        if epoch == cfg.unfreeze_after_epoch and cfg.freeze_encoder:
            print("\n" + "="*60)
            print(f"UNFREEZING ENCODER at epoch {epoch}")
            print("="*60)
            
            for param in model.encoder.parameters():
                param.requires_grad = True
            
            # Reduce batch size to avoid OOM when encoder gradients are enabled
            print("Reducing batch size from 2 to 1 to accommodate encoder gradients")
            print("Recreating dataloaders with batch_size=1...")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,  # Reduced from 2
                shuffle=True,
                num_workers=cfg.dataset_num_workers,
                collate_fn=collate_fn_classification,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,  # Reduced from 2
                shuffle=False,
                num_workers=cfg.dataset_num_workers,
                collate_fn=collate_fn_classification,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,  # Reduced from 2
                shuffle=False,
                num_workers=cfg.dataset_num_workers,
                collate_fn=collate_fn_classification,
                pin_memory=True
            )
            
            print("✓ Dataloaders recreated with batch_size=1")
            
            optimizer = torch.optim.AdamW([
                {'params': model.encoder.parameters(), 'lr': 0.0},
                {'params': model.classifier.parameters(), 'lr': cfg.base_lr}
            ], weight_decay=cfg.weight_decay)
            
            cfg._unfreeze_epoch = epoch
            cfg._encoder_lr_target = cfg.encoder_lr
    
    # Final test
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(cfg.checkpoint_dir, "best_model.pth"), 
                           map_location='cuda:0', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(-1, model, criterion, test_loader, cfg, pos_weights)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Val Acc: {best_val_acc:.3f}")
    print(f"Test Acc: {test_metrics['mean_accuracy']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
