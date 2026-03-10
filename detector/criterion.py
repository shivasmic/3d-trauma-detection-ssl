"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utils.dist import all_reduce_average
from utils.box_util import generalized_box3d_iou


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class Matcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_giou=2.0, cost_center=5.0, cost_size=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_giou = cost_giou
        self.cost_center = cost_center
        self.cost_size = cost_size

    @torch.no_grad()
    def forward(self, outputs, targets):
       
        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        pred_cls_prob = outputs["sem_cls_prob"].sigmoid()  
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (pred_cls_prob ** gamma) * (-(1 - pred_cls_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - pred_cls_prob) ** gamma) * (-(pred_cls_prob + 1e-8).log())
        class_mat = torch.gather(pos_cost_class - neg_cost_class, 2, gt_box_sem_cls_labels)

        center_mat = outputs["center_reg_dist"].detach()
        
        size_mat = outputs["size_reg_dist"].detach()
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_center * center_mat
            + self.cost_size * size_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class Criterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weights):
        super().__init__()
        self.matcher = matcher
        self.dataset_config = dataset_config
        self.loss_weights = loss_weights
        self.focal_alpha = 0.25
        
    def loss_sem_cls(self, outputs, targets, assignments):
        if targets["num_boxes_replica"] > 0:
            pred_logits = outputs["sem_cls_logits"]
            gt_box_label = torch.gather(
                targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
            )
            gt_box_label[assignments["proposal_matched_mask"].int() == 0] = pred_logits.shape[-1]

            target_classes_onehot = torch.zeros(
                [pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                dtype=pred_logits.dtype, 
                layout=pred_logits.layout, 
                device=pred_logits.device
            )
            target_classes_onehot.scatter_(2, gt_box_label.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]  
            
            loss = sigmoid_focal_loss(
                pred_logits, 
                target_classes_onehot,
                targets["num_boxes"], 
                alpha=self.focal_alpha, 
                gamma=2
            ) * pred_logits.shape[1]
        else:
            loss = outputs["sem_cls_logits"].sum() * 0.0

        return {"loss_sem_cls": loss}

    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_reg_dist"]

        if targets["num_boxes_replica"] > 0:
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = (outputs['center_unnormalized'].sum() * 0.0 + 
                          outputs.get('center_reg', outputs['center_unnormalized']).sum() * 0.0 +
                          outputs.get('pre_box_center_unnormalized', outputs['center_unnormalized']).sum() * 0.0)

        return {"loss_center": center_loss}

    def loss_size(self, outputs, targets, assignments):
        if targets["num_boxes_replica"] > 0:
            gt_box_sizes = targets["gt_box_sizes"]
            
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            
            if 'pre_box_size_unnormalized' in outputs and 'size_reg' in outputs:
                gt_size_reg = torch.log(
                    (gt_box_sizes + 1e-5) / (outputs["pre_box_size_unnormalized"] + 1e-5)
                )
                pred_size_reg = outputs["size_reg"]
                size_loss = F.l1_loss(gt_size_reg, pred_size_reg, reduction="none").sum(dim=-1)
            else:
                pred_size = outputs["size_unnormalized"]
                
                size_loss = F.l1_loss(
                    torch.log(pred_size + 1e-5), 
                    torch.log(gt_box_sizes + 1e-5), 
                    reduction="none"
                ).sum(dim=-1)
            
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = (outputs["size_unnormalized"].sum() * 0.0 +
                        outputs.get('size_reg', outputs['size_unnormalized']).sum() * 0.0 +
                        outputs.get('pre_box_size_unnormalized', outputs['size_unnormalized']).sum() * 0.0)

        return {"loss_size": size_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs["gious"]

        if targets["num_boxes_replica"] > 0:
            giou_loss = torch.gather(
                gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            
            giou_loss = giou_loss * assignments["proposal_matched_mask"]
            giou_loss = giou_loss.sum()

            if targets["num_boxes"] > 0:
                giou_loss /= targets["num_boxes"]
        else:
            giou_loss = outputs["size_unnormalized"].sum() * 0.0 + outputs["center_unnormalized"].sum() * 0.0

        return {"loss_giou": giou_loss}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        pred_logits = outputs["sem_cls_logits"]
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"].float())
        return {"loss_cardinality": card_err}

    def single_output_forward(self, outputs, targets):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=False,  
            needs_grad=(self.loss_weights["loss_giou_weight"] > 0),
        )
        outputs["gious"] = gious
        
        if 'pre_box_center_unnormalized' in outputs and 'center_reg' in outputs:
            gt_center_reg = (
                targets['gt_box_centers'].unsqueeze(1) - outputs['pre_box_center_unnormalized'].unsqueeze(2)
            ) / (outputs['pre_box_size_unnormalized'].unsqueeze(2) + 1e-5)
            outputs["center_reg_dist"] = torch.abs(
                outputs['center_reg'].unsqueeze(2) - gt_center_reg
            ).sum(-1)
        else:
            outputs["center_reg_dist"] = torch.abs(
                outputs['center_unnormalized'].unsqueeze(2) - targets['gt_box_centers'].unsqueeze(1)
            ).sum(-1)
        
        if 'pre_box_size_unnormalized' in outputs and 'size_reg' in outputs:
            gt_size_reg = torch.log(
                (targets['gt_box_sizes'].unsqueeze(1) + 1e-5) / 
                (outputs['pre_box_size_unnormalized'].unsqueeze(2) + 1e-5)
            )
            outputs["size_reg_dist"] = torch.abs(
                outputs['size_reg'].unsqueeze(2) - gt_size_reg
            ).sum(-1)
        else:
            outputs["size_reg_dist"] = torch.abs(
                torch.log(outputs['size_unnormalized'].unsqueeze(2) + 1e-5) - 
                torch.log(targets['gt_box_sizes'].unsqueeze(1) + 1e-5)
            ).sum(-1)
        
        assignments = self.matcher(outputs, targets)
        
        losses = {}
        
        loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            "loss_cardinality": self.loss_cardinality,
        }
        
        for loss_name, loss_fn in loss_functions.items():
            loss_wt_key = loss_name + "_weight"
            if loss_wt_key in self.loss_weights and self.loss_weights[loss_wt_key] > 0:
                curr_loss = loss_fn(outputs, targets, assignments)
                losses.update(curr_loss)
            elif loss_wt_key not in self.loss_weights:
                curr_loss = loss_fn(outputs, targets, assignments)
                losses.update(curr_loss)
        
        final_loss = 0
        for k in self.loss_weights:
            if self.loss_weights[k] > 0 and k.replace("_weight", "") in losses:
                losses[k.replace("_weight", "")] *= self.loss_weights[k]
                final_loss += losses[k.replace("_weight", "")]
        
        return final_loss, losses

    def forward(self, outputs, targets):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets["num_boxes_replica"] = nactual_gt.sum().item()
        
        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets)
        
        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets
                )
                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        
        return loss, loss_dict


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_size=args.matcher_size_cost,
    )

    loss_weights = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
    }
    
    criterion = Criterion(matcher, dataset_config, loss_weights)
    return criterion
