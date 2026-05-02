import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2 * inter + eps) / (union + eps)).mean()


def iou_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


def hausdorff_distance_metric(
    logits: torch.Tensor,
    targets: torch.Tensor,
    image_size: int,
    threshold: float = 0.5,
) -> float:
    """Symmetric Hausdorff distance averaged over the batch (in pixels).

    Empty/empty pair is treated as a perfect match (hd=0). Empty/non-empty
    pair is treated as the worst case (image diagonal).
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_np = preds.cpu().numpy().squeeze(1)
    targets_np = targets.cpu().numpy().squeeze(1)

    max_dist = float(np.sqrt(2 * image_size ** 2))
    batch_hd = []
    for i in range(preds_np.shape[0]):
        pred_coords = np.argwhere(preds_np[i])
        target_coords = np.argwhere(targets_np[i])
        if pred_coords.size == 0 and target_coords.size == 0:
            hd = 0.0
        elif pred_coords.size == 0 or target_coords.size == 0:
            hd = max_dist
        else:
            d1 = directed_hausdorff(pred_coords, target_coords)[0]
            d2 = directed_hausdorff(target_coords, pred_coords)[0]
            hd = max(d1, d2)
        batch_hd.append(hd)
    return float(np.mean(batch_hd))
