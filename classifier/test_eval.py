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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.dataset.rsna_classification_dataset import RSNAClassificationDataset, collate_fn_classification
from classifier.models.classification_model import build_classification_model
from classifier.models.unet_encoder import load_pretrained_unet_encoder


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for section, params in config_dict.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    setattr(self, key, value)


def evaluate_test_set(model, test_loader, device='cuda'):
    model.eval()
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    print("Running test evaluation...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            volumes = batch['volumes'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(volumes)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            if i % 50 == 0:
                print(f"  Processed {i}/{len(test_loader)} batches...")
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
    label_names = [
        'bowel_healthy', 'bowel_injury', 'liver_healthy', 'liver_high',
        'kidney_high', 'spleen_healthy', 'extravasation_injury'
    ]
    
    print("TEST SET RESULTS")
    print(f"Total test samples: {len(all_labels)}\n")
    
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
    
    if mean_auc > 0:
        print(f"OVERALL | Acc: {mean_acc:.3f} | AUC: {mean_auc:.3f}")
    else:
        print(f"OVERALL | Acc: {mean_acc:.3f}")
    
    return {
        'mean_accuracy': mean_acc,
        'mean_auc': mean_auc,
        'per_class_acc': all_accs,
        'per_class_auc': all_aucs,
        'label_names': label_names
    }


def main():
    print("CLASSIFICATION MODEL - TEST EVALUATION")
    
    # Load config
    config_path = 'configs/config.yaml'
    cfg = Config(config_path)
    
    print(f"\nLoading configuration from: {config_path}")
    print(f"Data directory: {cfg.data_dir}")
    print(f"CSV path: {cfg.csv_path}")
    print(f"Checkpoint: checkpoints/best_model.pth")
    
    # Create test dataset
    print("LOADING TEST DATASET")
    
    test_dataset = RSNAClassificationDataset(
        data_dir=cfg.data_dir,
        csv_path=cfg.csv_path,
        train_images_dir=cfg.train_images_dir,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.dataset_num_workers,
        collate_fn=collate_fn_classification,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print("LOADING MODEL")
    
    unet_encoder = load_pretrained_unet_encoder(
        pretrained_path=cfg.pretrained_weights,
        freeze_encoder=True,
        device='cuda:0'
    )
    
    model = build_classification_model(
        unet_encoder, 
        num_classes=cfg.num_classes,
        dropout=cfg.dropout if hasattr(cfg, 'dropout') else 0.5
    )
    model = model.cuda()
    
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Validation Acc: {checkpoint['metrics']['mean_accuracy']:.3f}")
    print(f"  Validation AUC: {checkpoint['metrics']['mean_auc']:.3f}")
    
    # Evaluate on test set
    test_metrics = evaluate_test_set(model, test_loader, device='cuda')
    
    # Final summary
    print("FINAL SUMMARY")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation Acc: {checkpoint['metrics']['mean_accuracy']*100:.2f}%")
    print(f"Validation AUC: {checkpoint['metrics']['mean_auc']*100:.2f}%")
    print(f"Test Acc: {test_metrics['mean_accuracy']*100:.2f}%")
    print(f"Test AUC: {test_metrics['mean_auc']*100:.2f}%")
    
    # Save results
    results = {
        'checkpoint_epoch': checkpoint['epoch'],
        'val_accuracy': checkpoint['metrics']['mean_accuracy'],
        'val_auc': checkpoint['metrics']['mean_auc'],
        'test_accuracy': test_metrics['mean_accuracy'],
        'test_auc': test_metrics['mean_auc'],
        'per_class_results': dict(zip(test_metrics['label_names'], 
                                     zip(test_metrics['per_class_acc'], 
                                         test_metrics['per_class_auc'])))
    }
    
    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to test_results.json")

if __name__ == "__main__":
    main()
