"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.rsna_dataset import RSNATraumaDataset, collate_fn
from dataset.rsna_target_preparation import RSNADatasetConfig, prepare_targets_rsna
from models.model_vdetr_unet import build_vdetr_unet
from models.unet_encoder import load_pretrained_unet_encoder
from utils.ap_calculator import RSNAAPCalculator


def make_args_parser():
    parser = argparse.ArgumentParser("V-DETR Test Evaluation", add_help=False)

    parser.add_argument("--test_data_dir", type=str, required=True,
                       help="Directory with held-out test .npz files")
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)

    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (best_model.pth)")
    parser.add_argument("--unet_checkpoint", type=str, 
                       default="../unet_liver_encoder_ckpt/3D_unet_liver_ssl_weights.pth",
                       help="Path to pretrained UNet checkpoint")

    parser.add_argument("--unet_channels", default=256, type=int)
    parser.add_argument("--unet_resolution", default=[32, 21, 21], nargs='+', type=int)
    parser.add_argument("--max_voxels", default=4096, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_nlayers", default=9, type=int)
    parser.add_argument("--dec_ffn_dim", default=1024, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)
    parser.add_argument("--dec_norm", default="ln", type=str)
    parser.add_argument("--rpe_dim", default=128, type=int)
    parser.add_argument("--rpe_quant", default="bilinear_4_10", type=str)
    parser.add_argument("--log_scale", default=512, type=float)
    parser.add_argument("--angle_type", default="", type=str)
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--querypos_mlp", default=False, action="store_true")
    parser.add_argument("--q_content", default="sample", type=str)
    parser.add_argument("--pos_for_key", default=False, action="store_true")
    parser.add_argument("--share_selfattn", default=False, action="store_true")
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument("--mlp_norm", default="bn1d", type=str)
    parser.add_argument("--mlp_act", default="relu", type=str)
    parser.add_argument("--mlp_sep", default=True, action="store_true")
    parser.add_argument("--is_bilable", default=True, action="store_true")
    parser.add_argument("--cls_loss", default="focalloss_0.25", type=str)

    parser.add_argument("--ap_iou_thresholds", default=[0.10, 0.25, 0.5, 0.75], nargs='+', type=float)
    parser.add_argument("--conf_thresh", default=0.05, type=float)
    parser.add_argument("--nms_iou", default=0.25, type=float)
    parser.add_argument("--use_nms", default=True, action="store_true")

    parser.add_argument("--output_dir", type=str, default="test_results",
                       help="Directory to save test results")
    parser.add_argument("--save_predictions", default=True, action="store_true",
                       help="Save per-sample predictions to JSON")

    return parser


def load_model(args, device='cuda'):
    
    print("LOADING MODEL")
    
    dataset_config = RSNADatasetConfig()
    
    print(f"Loading UNet encoder from: {args.unet_checkpoint}")
    if not os.path.exists(args.unet_checkpoint):
        raise FileNotFoundError(f"UNet checkpoint not found: {args.unet_checkpoint}")
    
    unet_encoder = load_pretrained_unet_encoder(
        pretrained_path=args.unet_checkpoint,
        freeze_encoder=False,
        device=device
    )
    
    # Build V-DETR model
    model = build_vdetr_unet(args, dataset_config, unet_encoder)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_val_metrics' in checkpoint:
        print(f"  Best val metrics: {checkpoint['best_val_metrics']}")
        
    return model, dataset_config


def evaluate_test_set(args, model, dataset_config, dataloader, device):
    model.eval()
    
    ap_calculator = RSNAAPCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=args.ap_iou_thresholds,
        conf_thresh=args.conf_thresh,
        nms_iou=args.nms_iou,
        use_nms=args.use_nms,
        class2type_map={0: 'background', 1: 'injury'},
    )
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], tuple):
                    batch[key] = tuple(x.to(device) if isinstance(x, torch.Tensor) else x 
                                      for x in batch[key])
            
            targets = prepare_targets_rsna(batch, dataset_config, device=device)
            
            outputs = model({
                'dicom_volumes': batch['dicom_volumes'],
                'volume_dims': batch['volume_dims'],
            })
            
            if args.cls_loss.startswith('focalloss'):
                outputs['outputs']['sem_cls_prob'] = outputs['outputs']['sem_cls_prob'].sigmoid()
            
            ap_calculator.step_meter(outputs, targets)
            
            if args.save_predictions:
                batch_size = batch['dicom_volumes'].shape[0]
                
                for i in range(batch_size):
                    pred_center = outputs['outputs']['center_unnormalized'][i, 0].cpu().numpy().tolist()
                    pred_size = outputs['outputs']['size_unnormalized'][i, 0].cpu().numpy().tolist()
                    pred_conf = outputs['outputs']['sem_cls_prob'][i, 0, 0].cpu().item()
                    
                    if isinstance(batch['series_ids'], (list, tuple)):
                        series_id = batch['series_ids'][i]
                    else:
                        series_id = batch['series_ids'].item() if batch_size == 1 else f"sample_{i}"
                    
                    try:
                        if i in targets:
                            gt_data = targets[i]
                        else:
                            gt_keys = list(targets.keys())
                            gt_data = targets[gt_keys[i]] if i < len(gt_keys) else None
                        
                        if gt_data is not None:
                            center_label = gt_data['center_label']
                            size_label = gt_data['size_label']
                            
                            while center_label.dim() > 1:
                                center_label = center_label[0]
                            while size_label.dim() > 1:
                                size_label = size_label[0]
                            
                            gt_center = center_label.cpu().numpy().tolist()
                            gt_size = size_label.cpu().numpy().tolist()
                        else:
                            gt_center = None
                            gt_size = None
                    except:
                        gt_center = None
                        gt_size = None
                    
                    sample_pred = {
                        'series_id': series_id,
                        'pred_center': pred_center,
                        'pred_size': pred_size,
                        'pred_confidence': pred_conf,
                        'gt_center': gt_center,
                        'gt_size': gt_size,
                    }
                    all_predictions.append(sample_pred)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Compute metrics
    print("\nComputing metrics...")
    ap_metrics_dict = ap_calculator.compute_metrics()
    ap_metrics_flat = ap_calculator.metrics_to_dict(ap_metrics_dict)
    ap_str = ap_calculator.metrics_to_str(ap_metrics_dict, per_class=True)
    
    print("TEST SET RESULTS")
    print(ap_str)
    
    return ap_metrics_flat, all_predictions


def save_results(args, metrics, predictions):    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")
    
    if args.save_predictions and predictions:
        pred_file = os.path.join(args.output_dir, 'test_predictions.json')
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"✓ Saved predictions to: {pred_file}")
    
    summary_file = os.path.join(args.output_dir, 'test_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("V-DETR LIVER DETECTION - TEST SET EVALUATION\n")
        f.write(f"Test Data Directory: {args.test_data_dir}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of test samples: {len(predictions)}\n\n")
        f.write("METRICS:\n")
        for key, value in sorted(metrics.items()):
            f.write(f"{key:30s}: {value:.4f}\n")
    
    print(f"Saved summary to: {summary_file}")
    print(f"\nAll results saved to: {args.output_dir}/")


def main(args):
    
    print("V-DETR LIVER DETECTION - TEST SET EVALUATION")
    print(f"Test data: {args.test_data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("LOADING TEST DATASET")
    
    test_dataset = RSNATraumaDataset(
        args.test_data_dir,
        split=None,             
    )
    
    print(f"Loaded {len(test_dataset)} test samples")
    print(f"Expected: 25 held-out liver volumes")
    
    if len(test_dataset) == 0:
        print("\n ERROR: No test samples found!")
        print(f"   Check that {args.test_data_dir} contains *_liver.npz files")
        sys.exit(1)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batchsize_per_gpu,
        shuffle=False,
        num_workers=args.dataset_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    model, dataset_config = load_model(args, device=device)
    
    metrics, predictions = evaluate_test_set(args, model, dataset_config, test_loader, device)
    
    save_results(args, metrics, predictions)
    
    print("TEST EVALUATION COMPLETE!")
    print(f"Results directory: {args.output_dir}")


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    main(args)
