"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch


def build_optimizer(args, model):

    params_with_decay = []
    params_without_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad is False:
            continue
        if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    if args.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.base_lr)
    return optimizer
