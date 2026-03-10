"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl

RSNA-specific AP Calculator for trauma detection.
Simplified from V-DETR's APCalculator for medical imaging with axis-aligned boxes.
"""


import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


def compute_iou_3d_batch(pred_corners: np.ndarray, gt_corners: np.ndarray) -> np.ndarray:
    
    N = pred_corners.shape[0]
    M = gt_corners.shape[0]
    ious = np.zeros((N, M))
    
    for i in range(N):
        pred_min = pred_corners[i].min(axis=0)  
        pred_max = pred_corners[i].max(axis=0)  
        pred_volume = np.prod(pred_max - pred_min + 1e-8)
        
        for j in range(M):
            gt_min = gt_corners[j].min(axis=0)  
            gt_max = gt_corners[j].max(axis=0)  
            gt_volume = np.prod(gt_max - gt_min + 1e-8)
            
            inter_min = np.maximum(pred_min, gt_min)
            inter_max = np.minimum(pred_max, gt_max)
            inter_dims = np.maximum(inter_max - inter_min, 0.0)
            intersection = np.prod(inter_dims)
            
            union = pred_volume + gt_volume - intersection
            ious[i, j] = intersection / (union + 1e-8)
    
    return ious


def parse_rsna_predictions(
    predicted_box_corners: torch.Tensor,
    sem_cls_probs: torch.Tensor,
    objectness_probs: torch.Tensor,
    conf_thresh: float = 0.05,
    nms_iou: float = 0.25,
    use_nms: bool = True,
) -> List[List[Tuple]]:
    
    pred_corners = predicted_box_corners.detach().cpu().numpy()  
    cls_probs = sem_cls_probs.detach().cpu().numpy()  
    obj_probs = objectness_probs.detach().cpu().numpy()  
    
    batch_size = pred_corners.shape[0]
    num_queries = pred_corners.shape[1]
    num_classes = cls_probs.shape[2]
    
    batch_pred_list = []
    
    for b in range(batch_size):
        if num_classes == 1:
            injury_cls_probs = cls_probs[b, :, 0]  
        elif num_classes == 2:
            injury_cls_probs = cls_probs[b, :, 1]  
        else:
            raise ValueError(
                f"Unexpected num_classes: {num_classes}. "
                f"Expected 1 (binary focal loss) or 2 (multi-class)."
            )
        
        scores = obj_probs[b] * injury_cls_probs  
        
        valid_mask = scores > conf_thresh
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            batch_pred_list.append([])
            continue
        
        valid_scores = scores[valid_indices]
        valid_corners = pred_corners[b, valid_indices]  
        
        sorted_indices = np.argsort(valid_scores)[::-1]
        valid_scores = valid_scores[sorted_indices]
        valid_corners = valid_corners[sorted_indices]
        
        if use_nms and len(valid_scores) > 1:
            keep_indices = []
            for i in range(len(valid_scores)):
                should_keep = True
                for kept_idx in keep_indices:
                    iou = compute_iou_3d_batch(
                        valid_corners[i:i+1],  
                        valid_corners[kept_idx:kept_idx+1]  
                    )[0, 0]
                    
                    if iou > nms_iou:
                        should_keep = False
                        break
                
                if should_keep:
                    keep_indices.append(i)
            
            valid_scores = valid_scores[keep_indices]
            valid_corners = valid_corners[keep_indices]
        
        pred_list = [
            (1, valid_corners[i], valid_scores[i])  
            for i in range(len(valid_scores))
        ]
        
        batch_pred_list.append(pred_list)
    
    return batch_pred_list


def parse_rsna_ground_truth(
    gt_box_corners: torch.Tensor,
    gt_box_present: torch.Tensor,
) -> List[List[Tuple]]:
   
    gt_corners = gt_box_corners.detach().cpu().numpy()
    gt_present = gt_box_present.detach().cpu().numpy()
    
    batch_size = gt_corners.shape[0]
    batch_gt_list = []
    
    for b in range(batch_size):
        valid_mask = gt_present[b] == 1
        valid_corners = gt_corners[b, valid_mask]  
        
        gt_list = [
            (1, valid_corners[i])  
            for i in range(len(valid_corners))
        ]
        
        batch_gt_list.append(gt_list)
    
    return batch_gt_list


def compute_ap_for_class(
    pred_list: List[Tuple],  
    gt_list: List[Tuple],     
    iou_thresh: float = 0.5,
    class_id: int = 1,
) -> Tuple[float, float, List[float], List[float]]:
   
    pred_class = [(corners, score) for (cls, corners, score) in pred_list if cls == class_id]
    gt_class = [corners for (cls, corners) in gt_list if cls == class_id]
    
    num_gt = len(gt_class)
    
    if num_gt == 0:
        return 0.0, 0.0, [], []
    
    if len(pred_class) == 0:
        return 0.0, 0.0, [0.0], [0.0]
    
    pred_class = sorted(pred_class, key=lambda x: x[1], reverse=True)
    
    gt_matched = [False] * num_gt
    
    tp = np.zeros(len(pred_class))
    fp = np.zeros(len(pred_class))
    
    for pred_idx, (pred_corners, score) in enumerate(pred_class):
        if num_gt > 0:
            ious = compute_iou_3d_batch(
                pred_corners[np.newaxis, ...],
                np.array(gt_class)              
            )[0]  
            
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]
        else:
            best_iou = 0.0
            best_gt_idx = -1
        
        if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    ap = 0.0
    for recall_thresh in np.linspace(0, 1, 11):
        precisions_above_thresh = precisions[recalls >= recall_thresh]
        if len(precisions_above_thresh) > 0:
            ap += precisions_above_thresh.max()
    ap /= 11.0
    
    ar = recalls[-1] if len(recalls) > 0 else 0.0
    
    return ap, ar, precisions.tolist(), recalls.tolist()


class RSNAAPCalculator:
    
    def __init__(
        self,
        dataset_config,
        ap_iou_thresh: List[float] = [0.25, 0.5, 0.75],
        conf_thresh: float = 0.05,
        nms_iou: float = 0.25,
        use_nms: bool = True,
        class2type_map: Optional[Dict] = None,
    ):
        self.dataset_config = dataset_config
        self.ap_iou_thresh = ap_iou_thresh
        self.conf_thresh = conf_thresh
        self.nms_iou = nms_iou
        self.use_nms = use_nms
        self.class2type_map = class2type_map or {0: 'background', 1: 'injury'}
        
        self.reset()
    
    def reset(self):
        self.all_preds = []  
        self.all_gts = []   
        self.scan_cnt = 0
    
    def step_meter(self, outputs: Dict, targets: Dict):
        if 'outputs' in outputs:
            outputs = outputs['outputs']
        
        batch_pred_list = parse_rsna_predictions(
            predicted_box_corners=outputs['box_corners'],
            sem_cls_probs=outputs['sem_cls_prob'],
            objectness_probs=outputs['objectness_prob'],
            conf_thresh=self.conf_thresh,
            nms_iou=self.nms_iou,
            use_nms=self.use_nms,
        )
        
        batch_gt_list = parse_rsna_ground_truth(
            gt_box_corners=targets['gt_box_corners'],
            gt_box_present=targets['gt_box_present'],
        )
        
        for preds, gts in zip(batch_pred_list, batch_gt_list):
            self.all_preds.extend(preds)
            self.all_gts.extend(gts)
            self.scan_cnt += 1
    
    def compute_metrics(self) -> OrderedDict:
        overall_ret = OrderedDict()
        
        for iou_thresh in self.ap_iou_thresh:
            ret_dict = OrderedDict()
            
            ap, ar, prec_curve, rec_curve = compute_ap_for_class(
                pred_list=self.all_preds,
                gt_list=self.all_gts,
                iou_thresh=iou_thresh,
                class_id=1,
            )
            
            class_name = self.class2type_map[1]
            ret_dict[f"{class_name} Average Precision"] = ap
            ret_dict[f"{class_name} Recall"] = ar
            ret_dict["mAP"] = ap  
            ret_dict["AR"] = ar
            
            overall_ret[iou_thresh] = ret_dict
        
        return overall_ret
    
    def metrics_to_str(self, overall_ret: OrderedDict, per_class: bool = True) -> str:
        mAP_strs = []
        AR_strs = []
        per_class_metrics = []
        
        for iou_thresh in self.ap_iou_thresh:
            mAP = overall_ret[iou_thresh]["mAP"] * 100
            mAP_strs.append(f"{mAP:.2f}")
            ar = overall_ret[iou_thresh]["AR"] * 100
            AR_strs.append(f"{ar:.2f}")
            
            if per_class:
                per_class_metrics.append("-" * 40)
                per_class_metrics.append(f"IoU Threshold = {iou_thresh}")
                for key, value in overall_ret[iou_thresh].items():
                    if key not in ["mAP", "AR"]:
                        met_str = f"  {key}: {value * 100:.2f}%"
                        per_class_metrics.append(met_str)
        
        ap_header = [f"mAP@{x:.2f}" for x in self.ap_iou_thresh]
        result = ", ".join(ap_header) + ": " + ", ".join(mAP_strs) + "\n"
        
        ar_header = [f"AR@{x:.2f}" for x in self.ap_iou_thresh]
        result += ", ".join(ar_header) + ": " + ", ".join(AR_strs)
        
        if per_class:
            result += "\n" + "\n".join(per_class_metrics)
        
        return result
    
    def metrics_to_dict(self, overall_ret: OrderedDict) -> Dict:
        metrics_dict = {}
        for iou_thresh in self.ap_iou_thresh:
            metrics_dict[f"mAP_{iou_thresh}"] = overall_ret[iou_thresh]["mAP"] * 100
            metrics_dict[f"AR_{iou_thresh}"] = overall_ret[iou_thresh]["AR"] * 100
        return metrics_dict
    
    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)
