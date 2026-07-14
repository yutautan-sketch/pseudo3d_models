from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


LABEL_IGNORE = -1
LABEL_BACKGROUND = 0
LABEL_FEMUR = 1


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator.float() / denominator.float(),
        torch.zeros_like(numerator, dtype=torch.float32),
    )


def make_metric_mask(
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = LABEL_IGNORE,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, N], got {tuple(labels.shape)}")
    mask = labels != int(ignore_index)
    if valid_mask is not None:
        if valid_mask.shape != labels.shape:
            raise ValueError(
                f"valid_mask must have shape {tuple(labels.shape)}, got {tuple(valid_mask.shape)}"
            )
        mask = mask & valid_mask.bool()
    return mask


def predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, N, C], got {tuple(logits.shape)}")
    return torch.argmax(logits, dim=-1)


@dataclass
class BinarySegmentationCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    tn: torch.Tensor
    fn: torch.Tensor
    valid: torch.Tensor
    total: torch.Tensor
    positive: torch.Tensor

    @classmethod
    def empty(cls, *, device: torch.device | None = None) -> "BinarySegmentationCounts":
        kwargs: dict[str, Any] = {"dtype": torch.float64}
        if device is not None:
            kwargs["device"] = device
        zero = torch.zeros((), **kwargs)
        return cls(
            tp=zero.clone(),
            fp=zero.clone(),
            tn=zero.clone(),
            fn=zero.clone(),
            valid=zero.clone(),
            total=zero.clone(),
            positive=zero.clone(),
        )

    def add_(self, other: "BinarySegmentationCounts") -> None:
        self.tp += other.tp.to(self.tp.device)
        self.fp += other.fp.to(self.fp.device)
        self.tn += other.tn.to(self.tn.device)
        self.fn += other.fn.to(self.fn.device)
        self.valid += other.valid.to(self.valid.device)
        self.total += other.total.to(self.total.device)
        self.positive += other.positive.to(self.positive.device)

    def to_metrics(self) -> dict[str, float]:
        accuracy = _safe_divide(self.tp + self.tn, self.valid)
        precision = _safe_divide(self.tp, self.tp + self.fp)
        recall = _safe_divide(self.tp, self.tp + self.fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        iou_femur = _safe_divide(self.tp, self.tp + self.fp + self.fn)
        iou_background = _safe_divide(self.tn, self.tn + self.fp + self.fn)
        mean_iou = (iou_femur + iou_background) * 0.5
        valid_ratio = _safe_divide(self.valid, self.total)
        positive_ratio = _safe_divide(self.positive, self.valid)

        return {
            "accuracy": float(accuracy.item()),
            "precision": float(precision.item()),
            "recall": float(recall.item()),
            "f1": float(f1.item()),
            "iou_femur": float(iou_femur.item()),
            "iou_background": float(iou_background.item()),
            "mean_iou": float(mean_iou.item()),
            "valid_ratio": float(valid_ratio.item()),
            "positive_ratio": float(positive_ratio.item()),
            "valid_point_count": float(self.valid.item()),
            "total_point_count": float(self.total.item()),
        }


def compute_binary_counts(
    pred_label: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = LABEL_IGNORE,
    positive_label: int = LABEL_FEMUR,
) -> BinarySegmentationCounts:
    if pred_label.shape != labels.shape:
        raise ValueError(
            f"pred_label and labels must share shape, got {tuple(pred_label.shape)} "
            f"and {tuple(labels.shape)}"
        )

    metric_mask = make_metric_mask(labels, valid_mask, ignore_index=ignore_index)
    pred = pred_label[metric_mask]
    target = labels[metric_mask]

    pred_pos = pred == int(positive_label)
    target_pos = target == int(positive_label)
    pred_neg = ~pred_pos
    target_neg = ~target_pos

    counts = BinarySegmentationCounts.empty(device=labels.device)
    counts.tp = (pred_pos & target_pos).sum(dtype=torch.float64)
    counts.fp = (pred_pos & target_neg).sum(dtype=torch.float64)
    counts.tn = (pred_neg & target_neg).sum(dtype=torch.float64)
    counts.fn = (pred_neg & target_pos).sum(dtype=torch.float64)
    counts.valid = metric_mask.sum(dtype=torch.float64)
    counts.total = torch.as_tensor(labels.numel(), device=labels.device, dtype=torch.float64)
    counts.positive = target_pos.sum(dtype=torch.float64)
    return counts


def compute_segmentation_metrics(
    output: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    ignore_index: int = LABEL_IGNORE,
    positive_label: int = LABEL_FEMUR,
) -> dict[str, float]:
    if "logits" not in output:
        raise KeyError("Model output must contain 'logits'")
    if "labels" not in batch:
        raise KeyError("Batch must contain 'labels'")

    pred_label = predictions_from_logits(output["logits"])
    counts = compute_binary_counts(
        pred_label,
        batch["labels"],
        batch.get("valid_mask"),
        ignore_index=ignore_index,
        positive_label=positive_label,
    )
    return counts.to_metrics()


class SegmentationMetricAccumulator:
    """Accumulate point-wise binary segmentation metrics across batches."""

    def __init__(
        self,
        *,
        ignore_index: int = LABEL_IGNORE,
        positive_label: int = LABEL_FEMUR,
        device: torch.device | None = None,
    ) -> None:
        self.ignore_index = int(ignore_index)
        self.positive_label = int(positive_label)
        self.counts = BinarySegmentationCounts.empty(device=device)

    def reset(self) -> None:
        device = self.counts.tp.device
        self.counts = BinarySegmentationCounts.empty(device=device)

    @torch.no_grad()
    def update(self, output: dict[str, torch.Tensor], batch: dict[str, Any]) -> None:
        pred_label = predictions_from_logits(output["logits"])
        counts = compute_binary_counts(
            pred_label,
            batch["labels"],
            batch.get("valid_mask"),
            ignore_index=self.ignore_index,
            positive_label=self.positive_label,
        )
        self.counts.add_(counts)

    def compute(self) -> dict[str, float]:
        return self.counts.to_metrics()
