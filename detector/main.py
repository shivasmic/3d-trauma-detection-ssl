"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import argparse
import os
import sys
import yaml

import numpy as np
import torch
from criterion import build_criterion
from optimizer import build_optimizer
from utils.ap_calculator import RSNAAPCalculator
from models.model_vdetr_unet import build_vdetr_unet
from utils.io import save_checkpoint, resume_if_possible
from torch.utils.data import DataLoader, DistributedSampler
from models.unet_encoder import load_pretrained_unet_encoder
from dataset.rsna_dataset import RSNATraumaDataset, collate_fn
from loss.consistency_loss import ConsistencyLoss, get_consistency_weight
from utils.dist import init_distributed, is_distributed, is_primary, get_rank
from dataset.rsna_unlabeled_dataset import RSNAUnlabeledDataset, collate_fn_unlabeled
from dataset.rsna_target_preparation import RSNADatasetConfig, prepare_targets_rsna





def make_args_parser():
    parser = argparse.ArgumentParser("RSNA V-DETR Training", add_help=False)

    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config YAML file")

    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory with preprocessed .npz files")
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)

    parser.add_argument("--unet_checkpoint", type=str, default="../unet_weights/v2/3D_unet_ssl_weights.pth",
                       help="Path to pretrained UNet checkpoint (SSL weights)")
    parser.add_argument("--freeze_unet", default=True, type=bool,
                       help="Freeze UNet encoder initially")
    parser.add_argument("--unfreeze_after_epoch", default=20, type=int,
                       help="Unfreeze UNet encoder after this epoch")
    parser.add_argument("--unfreeze_warmup_epochs", default=3, type=int,
                       help="Ramp encoder LR from 0 to encoder_lr over this many epochs after unfreeze")

    parser.add_argument("--unet_channels", default=256, type=int,
                       help="Output channels from UNet encoder")
    parser.add_argument("--unet_resolution", default=[32, 21, 21], nargs='+', type=int,
                       help="Feature resolution after UNet encoding")
    parser.add_argument("--max_voxels", default=4096, type=int,
                       help="Max voxels to sample for transformer")

    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_nlayers", default=9, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
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
    parser.add_argument("--loss_giou_weight", default=2.0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=3.0, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    parser.add_argument("--matcher_cls_cost", default=2.0, type=float)
    parser.add_argument("--matcher_giou_cost", default=2.0, type=float)
    parser.add_argument("--matcher_center_cost", default=5.0, type=float)
    parser.add_argument("--matcher_size_cost", default=1.0, type=float)

    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--encoder_lr", default=1e-5, type=float,
                       help="LR for UNet encoder after unfreeze (should be << base_lr)")
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--clip_gradient", default=0.1, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--warm_lr_epochs", default=5, type=int)
    parser.add_argument("--filter_biases_wd", default=True, action="store_true",
                   help="Filter out biases and 1D params from weight decay")

    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--eval_every_epoch", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    parser.add_argument("--ap_iou_thresholds", default=[0.10, 0.25, 0.5, 0.75], nargs='+', type=float,
                       help="IoU thresholds for AP computation")
    parser.add_argument("--conf_thresh", default=0.05, type=float,
                       help="Confidence threshold for predictions")
    parser.add_argument("--nms_iou", default=0.25, type=float,
                       help="IoU threshold for NMS")
    parser.add_argument("--use_nms", default=True, action="store_true",
                       help="Whether to use NMS during evaluation")

    parser.add_argument("--ssl_enabled", default=True, type=bool,
                       help="Enable semi-supervised consistency training")
    parser.add_argument("--ssl_start_epoch", default=0, type=int,
                       help="Epoch when SSL consistency loss starts")
    parser.add_argument("--ssl_unlabeled_data_dir", default=None, type=str,
                       help="Directory with unlabeled .npz files (defaults to --data_dir)")
    parser.add_argument("--ssl_unlabeled_batch_size", default=2, type=int,
                       help="Batch size for unlabeled dataloader")
    parser.add_argument("--ssl_consistency_center_weight", default=1.0, type=float)
    parser.add_argument("--ssl_consistency_size_weight", default=1.0, type=float)
    parser.add_argument("--ssl_consistency_cls_weight", default=1.0, type=float)
    parser.add_argument("--ssl_consistency_max_weight", default=1.0, type=float,
                       help="Peak multiplier on total consistency loss after warmup")
    parser.add_argument("--ssl_consistency_warmup_epochs", default=10, type=int,
                       help="Epochs to linearly ramp consistency weight from 0")
    parser.add_argument("--ssl_temperature", default=2.0, type=float,
                       help="Temperature for soft targets (reserved for future multi-class KL)")

    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=5, type=int)

    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    parser.add_argument("--wandb_activate", default=False, type=bool)
    parser.add_argument("--wandb_entity", default=None, type=str)
    parser.add_argument("--wandb_project", default="rsna-vdetr", type=str)
    parser.add_argument("--wandb_key", default="", type=str)

    return parser


def load_config(config_path, args):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for section, params in config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    setattr(args, key, value)

        print(f"Loaded config from {config_path}")
        print(f"Key settings: nqueries={args.nqueries}, dec_nlayers={args.dec_nlayers}, ssl_start_epoch={args.ssl_start_epoch}")
    else:
        print(f"Config file {config_path} not found, using command line args only")


def train_one_epoch(args, epoch, model, optimizer, criterion, dataset_config,
                    dataloader, unlabeled_loader=None, consistency_loss_fn=None):
    model.train()
    criterion.train()

    total_loss = 0
    loss_details = {}

    ssl_active = (unlabeled_loader is not None and consistency_loss_fn is not None
                  and getattr(args, 'ssl_enabled', False))
    if ssl_active:
        unlabeled_iter = iter(unlabeled_loader)
        consistency_weight = get_consistency_weight(
            epoch,
            start_epoch=args.ssl_start_epoch,
            warmup_epochs=args.ssl_consistency_warmup_epochs,
            max_weight=args.ssl_consistency_max_weight,
        )
        if is_primary():
            print(f"[SSL] epoch={epoch}  consistency_weight={consistency_weight:.4f}")

    for batch_idx, batch in enumerate(dataloader):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
            elif isinstance(batch[key], tuple):
                batch[key] = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in batch[key])

        targets = prepare_targets_rsna(batch, dataset_config, device=torch.device('cuda'))

        outputs = model({
            'dicom_volumes': batch['dicom_volumes'],
            'volume_dims': batch['volume_dims'],
        })

        loss, loss_dict = criterion(outputs, targets)

        if ssl_active and consistency_weight > 0:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            for key in unlabeled_batch:
                if isinstance(unlabeled_batch[key], torch.Tensor):
                    unlabeled_batch[key] = unlabeled_batch[key].cuda()
                elif isinstance(unlabeled_batch[key], tuple):
                    unlabeled_batch[key] = tuple(
                        x.cuda() if isinstance(x, torch.Tensor) else x
                        for x in unlabeled_batch[key]
                    )

            with torch.no_grad():
                weak_outputs = model({
                    'dicom_volumes': unlabeled_batch['weak_volume'],
                    'volume_dims':   unlabeled_batch['volume_dims'],
                })

            strong_outputs = model({
                'dicom_volumes': unlabeled_batch['strong_volume'],
                'volume_dims':   unlabeled_batch['volume_dims'],
            })

            con_loss, con_loss_detail = consistency_loss_fn(weak_outputs, strong_outputs)
            con_loss = con_loss * consistency_weight
            loss = loss + con_loss

            for k, v in con_loss_detail.items():
                loss_dict[k] = v * consistency_weight

        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in loss_details:
                loss_details[key] = 0
            loss_details[key] += value.item() if isinstance(value, torch.Tensor) else value

        if batch_idx % args.log_every == 0 and is_primary():
            log_msg = (
                f"Epoch [{epoch}/{args.max_epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"Cls: {loss_dict.get('loss_sem_cls', 0):.4f} "
                f"Center: {loss_dict.get('loss_center', 0):.4f} "
                f"Size: {loss_dict.get('loss_size', 0):.4f} "
                f"GIoU: {loss_dict.get('loss_giou', 0):.4f}"
            )
            if ssl_active:
                log_msg += (
                    f" | SSL con: {loss_dict.get('loss_consistency', 0):.4f} "
                    f"[ctr={loss_dict.get('loss_consistency_center', 0):.4f} "
                    f"sz={loss_dict.get('loss_consistency_size', 0):.4f} "
                    f"cls={loss_dict.get('loss_consistency_cls', 0):.4f}]"
                )
            print(log_msg)

    avg_loss = total_loss / len(dataloader)
    for key in loss_details:
        loss_details[key] /= len(dataloader)

    return avg_loss, loss_details


def evaluate(args, epoch, model, criterion, dataset_config, dataloader):
    model.eval()
    if criterion is not None:
        criterion.eval()

    ap_calculator = RSNAAPCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=args.ap_iou_thresholds,
        conf_thresh=args.conf_thresh,
        nms_iou=args.nms_iou,
        use_nms=args.use_nms,
        class2type_map={0: 'background', 1: 'injury'},
    )

    total_loss = 0
    loss_details = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
                elif isinstance(batch[key], tuple):
                    batch[key] = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in batch[key])

            targets = prepare_targets_rsna(batch, dataset_config, device=torch.device('cuda'))

            outputs = model({
                'dicom_volumes': batch['dicom_volumes'],
                'volume_dims': batch['volume_dims'],
            })

            if criterion is not None:
                loss, loss_dict = criterion(outputs, targets)
                total_loss += loss.item()

                for key, value in loss_dict.items():
                    if key not in loss_details:
                        loss_details[key] = 0
                    loss_details[key] += value.item() if isinstance(value, torch.Tensor) else value

            if args.cls_loss.startswith('focalloss'):
                outputs['outputs']['sem_cls_prob'] = outputs['outputs']['sem_cls_prob'].sigmoid()

            ap_calculator.step_meter(outputs, targets)

            if batch_idx % args.log_every == 0 and is_primary():
                print(f"Validation Epoch [{epoch}/{args.max_epoch}] "
                      f"Batch [{batch_idx}/{len(dataloader)}]")

    avg_loss = total_loss / len(dataloader) if criterion is not None else 0
    if criterion is not None:
        for key in loss_details:
            loss_details[key] /= len(dataloader)

    ap_metrics_dict = ap_calculator.compute_metrics()
    ap_metrics_flat = ap_calculator.metrics_to_dict(ap_metrics_dict)
    ap_str = ap_calculator.metrics_to_str(ap_metrics_dict, per_class=True)

    if is_primary():
        print(f"VALIDATION RESULTS - Epoch [{epoch}/{args.max_epoch}]")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Classification: {loss_details.get('loss_sem_cls', 0):.4f}")
        print(f"Center:         {loss_details.get('loss_center', 0):.4f}")
        print(f"Size:           {loss_details.get('loss_size', 0):.4f}")
        print(f"GIoU:           {loss_details.get('loss_giou', 0):.4f}")
        print("\nDetection Metrics:")
        print(ap_str)

    return avg_loss, loss_details, ap_metrics_flat


def main(local_rank, args):
    if args.ngpus > 1:
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    torch.cuda.set_device(local_rank)

    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    if hasattr(args, 'config') and args.config:
        load_config(args.config, args)

    dataset_config = RSNADatasetConfig()

    if args.test_only:
        test_dataset = RSNATraumaDataset(args.data_dir, split='test', use_labeled_only=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batchsize_per_gpu,
            shuffle=False,
            num_workers=args.dataset_num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        dataloaders = {'test': test_loader}
        train_dataset = None
        val_dataset = test_dataset
    else:
        train_dataset = RSNATraumaDataset(args.data_dir, split='train', use_labeled_only=True)
        val_dataset = RSNATraumaDataset(args.data_dir, split='val', use_labeled_only=True)

        if is_distributed():
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        dataloaders = {
            'train': train_loader,
            'test': val_loader,
            'train_sampler': train_sampler,
            'test_sampler': val_sampler,
        }

    if not os.path.exists(args.unet_checkpoint):
        raise FileNotFoundError(
            f"UNet checkpoint not found: {args.unet_checkpoint}\n"
            f"Please ensure you have run SSL pre-training and saved weights at this path."
        )

    unet_encoder = load_pretrained_unet_encoder(
        pretrained_path=args.unet_checkpoint,
        freeze_encoder=args.freeze_unet,
        device=f'cuda:{local_rank}'
    )

    print("="*60 + "\n")

    model = build_vdetr_unet(args, dataset_config, unet_encoder)
    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    criterion = build_criterion(args, dataset_config)
    criterion = criterion.cuda(local_rank)

    ssl_active = getattr(args, 'ssl_enabled', False) and not args.test_only
    unlabeled_loader = None
    consistency_loss_fn = None

    if ssl_active:
        unlabeled_dir = getattr(args, 'ssl_unlabeled_data_dir', None) or args.data_dir

        unlabeled_dataset = RSNAUnlabeledDataset(unlabeled_dir)

        if is_distributed():
            unlabeled_sampler = DistributedSampler(unlabeled_dataset, shuffle=True)
        else:
            unlabeled_sampler = torch.utils.data.RandomSampler(unlabeled_dataset)

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            sampler=unlabeled_sampler,
            batch_size=args.ssl_unlabeled_batch_size,
            num_workers=args.dataset_num_workers,
            collate_fn=collate_fn_unlabeled,
            pin_memory=True,
        )

        consistency_loss_fn = ConsistencyLoss(
            center_weight=args.ssl_consistency_center_weight,
            size_weight=args.ssl_consistency_size_weight,
            cls_weight=args.ssl_consistency_cls_weight,
            temperature=args.ssl_temperature,
            start_epoch=args.ssl_start_epoch,                   
            warmup_epochs=args.ssl_consistency_warmup_epochs,    
            max_weight=args.ssl_consistency_max_weight
        ).cuda(local_rank)

        if is_primary():
            print("SEMI-SUPERVISED LEARNING ENABLED")
            print(f"Unlabeled volumes : {len(unlabeled_dataset)}")
            print(f"SSL start epoch   : {args.ssl_start_epoch}")
            print(f"Warmup epochs     : {args.ssl_consistency_warmup_epochs}")
            print(f"Max weight        : {args.ssl_consistency_max_weight}")

    if args.test_only:
        if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
            sys.exit(1)

        sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"), weights_only = False)
        model_no_ddp.load_state_dict(sd["model"], strict=False)

        val_loss, val_loss_details, ap_metrics = evaluate(
            args, -1, model, criterion, dataset_config, dataloaders['test']
        )
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        optimizer = build_optimizer(args, model_no_ddp)

        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1

        best_map = best_val_metrics.get('mAP_0.5', 0.0) if best_val_metrics else 0.0

        for epoch in range(args.start_epoch, args.max_epoch):
            if is_distributed():
                dataloaders['train_sampler'].set_epoch(epoch)

            train_loss, train_loss_details = train_one_epoch(
                args, epoch, model, optimizer, criterion, dataset_config, dataloaders['train'],
                unlabeled_loader=unlabeled_loader,
                consistency_loss_fn=consistency_loss_fn,
            )

            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics={'mAP_0.5': best_map},
                filename="checkpoint.pth",
            )

            if epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1):
                val_loss, val_loss_details, ap_metrics = evaluate(
                    args, epoch, model, criterion, dataset_config, dataloaders['test']
                )

                current_map = ap_metrics.get('mAP_0.5', 0.0)
                if current_map > best_map:
                    best_map = current_map
                    save_checkpoint(
                        args.checkpoint_dir,
                        model_no_ddp,
                        optimizer,
                        epoch,
                        args,
                        best_val_metrics={'mAP_0.5': best_map},
                        filename="best_model.pth",
                    )
                    if is_primary():
                        print(f"New best model saved! mAP@0.5: {best_map:.2f}")


            if epoch == args.unfreeze_after_epoch and args.freeze_unet:
                if is_primary():
                    print(f"UNFREEZING UNET ENCODER at epoch {epoch}")

                for name, param in model_no_ddp.unet_encoder.named_parameters():
                    if any(x in name for x in ['down', 'pool', 'bottleneck']):
                        param.requires_grad = True

                encoder_params = []
                other_params   = []
                encoder_param_ids = set(id(p) for p in model_no_ddp.unet_encoder.parameters())

                for p in model_no_ddp.parameters():
                    if not p.requires_grad:
                        continue
                    if id(p) in encoder_param_ids:
                        encoder_params.append(p)
                    else:
                        other_params.append(p)

                encoder_lr = getattr(args, 'encoder_lr', args.base_lr * 0.1)
                unfreeze_warmup = getattr(args, 'unfreeze_warmup_epochs', 3)

                optimizer = torch.optim.AdamW([
                    {'params': encoder_params, 'lr': 0.0,          'initial_lr': encoder_lr},
                    {'params': other_params,   'lr': args.base_lr, 'initial_lr': args.base_lr},
                ], weight_decay=args.weight_decay)

                args._unfreeze_epoch       = epoch
                args._encoder_lr_target    = encoder_lr
                args._unfreeze_warmup      = unfreeze_warmup
                args._optimizer_rebuilt    = True

                if is_primary():
                    trainable = sum(p.numel() for p in model_no_ddp.parameters() if p.requires_grad)
                    total     = sum(p.numel() for p in model_no_ddp.parameters())
                    print(f"Trainable parameters: {trainable:,} / {total:,}")
                    print(f"Encoder group : {len(encoder_params):,} params @ lr={encoder_lr}")
                    print(f"Decoder group : {len(other_params):,} params @ lr={args.base_lr}")

            if getattr(args, '_optimizer_rebuilt', False):
                epochs_since_unfreeze = epoch - args._unfreeze_epoch
                if epochs_since_unfreeze <= args._unfreeze_warmup:
                    ramp_frac = epochs_since_unfreeze / args._unfreeze_warmup
                    current_encoder_lr = args._encoder_lr_target * ramp_frac
                    optimizer.param_groups[0]['lr'] = current_encoder_lr
                    if is_primary():
                        print(f"Encoder LR ramp at epoch {epoch}  encoder_lr={current_encoder_lr:.2e}")


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()

    if args.ngpus == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=args.ngpus, args=(args,))
