import torch
import torch.nn.functional as F
from config import NUM_CLASSES


def dice_score(preds: torch.Tensor, targets: torch.Tensor, eps=1e-6) -> float:
    """
    Mean Dice Score across foreground classes (excludes background class 0).
    preds  : (B, C, H, W) logits  or  (B, H, W) class indices
    targets: (B, H, W) class indices (long)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)  # (B, H, W)

    dice_per_class = []
    for c in range(1, NUM_CLASSES):  # skip background
        pred_c   = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union < eps:
            continue  # class absent in this batch
        dice_per_class.append((2 * intersection + eps) / (union + eps))

    return torch.stack(dice_per_class).mean().item() if dice_per_class else 0.0


def dice_per_class(preds: torch.Tensor, targets: torch.Tensor, eps=1e-6) -> dict:
    """Return Dice score per foreground class."""
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    labels = {1: "RV", 2: "Myo", 3: "LV"}
    result = {}
    for c, name in labels.items():
        pred_c   = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        result[name] = ((2 * intersection + eps) / (union + eps)).item() if union >= eps else 0.0
    return result
