"""
Consistency loss for semi-supervised CT-VDETR training.
Teacher = same network on the weak view (torch.no_grad()); student = same network on the strong view.
Returns the RAW loss. lambda(t) is applied ONCE in main.py via get_consistency_weight().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCENE_SIZE_MM = (1024.0, 336.0, 336.0)


class ConsistencyLoss(nn.Module):
    def __init__(self, center_weight=1.0, size_weight=1.0, cls_weight=1.0,
                 temperature=2.0, **unused):  # old start_epoch/warmup_epochs/max_weight kwargs are ignored
        super().__init__()
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        self.register_buffer("scene_size", torch.tensor(_SCENE_SIZE_MM), persistent=False)

    def _normalized(self, pred, key):
        if f"{key}_normalized" in pred:
            return pred[f"{key}_normalized"]
        x = pred[f"{key}_unnormalized"]
        return x / self.scene_size.to(x.device)

    def forward(self, weak_outputs, strong_outputs, epoch=None):
        weak_pred = weak_outputs["outputs"]
        strong_pred = strong_outputs["outputs"]
        T = self.temperature

        loss_center = F.l1_loss(self._normalized(strong_pred, "center"),
                                self._normalized(weak_pred, "center").detach())
        loss_size = F.l1_loss(self._normalized(strong_pred, "size"),
                              self._normalized(weak_pred, "size").detach())

        key = "sem_cls_logits" if "sem_cls_logits" in weak_pred else "sem_cls_prob"
        weak_logits = weak_pred[key].detach()
        strong_logits = strong_pred[key]
        if strong_logits.shape[-1] == 1:
            target = torch.sigmoid(weak_logits / T)
            loss_cls = F.binary_cross_entropy_with_logits(strong_logits / T, target) * (T ** 2)
        else:
            weak_soft = F.softmax(weak_logits / T, dim=-1)
            strong_log_soft = F.log_softmax(strong_logits / T, dim=-1)
            loss_cls = F.kl_div(strong_log_soft, weak_soft, reduction="none").sum(-1).mean() * (T ** 2)

        total_loss = (self.center_weight * loss_center +
                      self.size_weight * loss_size +
                      self.cls_weight * loss_cls)
        return total_loss, {
            "loss_consistency": total_loss.item(),
            "loss_consistency_center": loss_center.item(),
            "loss_consistency_size": loss_size.item(),
            "loss_consistency_cls": loss_cls.item(),
        }


def get_consistency_weight(epoch, start_epoch=0, warmup_epochs=10, max_weight=1.0):
    if epoch < start_epoch:
        return 0.0
    return max_weight * min(1.0, (epoch - start_epoch) / max(warmup_epochs, 1))
