from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class AuxLossWeights:
    segmentation: float = 1.0
    heatmap: float = 1.0
    paf: float = 0.25


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _dist_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def build_aux_loss_weights(train_config: Any) -> AuxLossWeights:
    return AuxLossWeights(
        segmentation=float(_get(train_config, "W_AUX_SEG", 1.0)),
        heatmap=float(_get(train_config, "W_AUX_HEATMAP", 1.0)),
        paf=float(_get(train_config, "W_AUX_PAF", 0.25)),
    )


def _prepare_aux_batch(batchdata: Any, device: torch.device) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    batch = batchdata[0]
    images = [img.to(device, dtype=torch.float32) for img in batch[0]]
    targets = {
        "paf": batch[3].to(device, dtype=torch.float32),
        "paf_mask": batch[4].to(device, dtype=torch.bool),
        "segmentation": batch[5].to(device, dtype=torch.float32),
        "heatmap": batch[6].to(device, dtype=torch.float32),
    }
    return images, targets


def _resize_like(source: torch.Tensor, target: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    if mode == "nearest":
        return F.interpolate(source, size=target.shape[-2:], mode=mode)
    return F.interpolate(source, size=target.shape[-2:], mode=mode, align_corners=False)


def compute_aux_losses(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: AuxLossWeights,
) -> dict[str, torch.Tensor]:
    aux_maps = output.get("aux_maps")
    if aux_maps is None:
        raise KeyError("model output must contain 'aux_maps' for aux supervised training")
    if aux_maps.shape[1] < 4:
        raise ValueError(f"aux_maps must have at least 4 channels, got shape {tuple(aux_maps.shape)}")

    seg_target = targets["segmentation"]
    heatmap_target = targets["heatmap"]
    paf_target = targets["paf"]
    paf_mask = targets["paf_mask"].to(dtype=torch.float32)

    seg_logits = _resize_like(aux_maps[:, 0:1], seg_target)
    heatmap_logits = _resize_like(aux_maps[:, 1:2], heatmap_target)
    paf_pred = _resize_like(aux_maps[:, 2:4], paf_target)

    seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target)
    heatmap_mse = F.mse_loss(torch.sigmoid(heatmap_logits), heatmap_target)
    paf_l1 = (torch.abs(torch.tanh(paf_pred) - paf_target) * paf_mask).sum()
    paf_l1 = paf_l1 / (paf_mask.sum().clamp_min(1.0) * paf_target.shape[1])

    total = (
        weights.segmentation * seg_bce
        + weights.heatmap * heatmap_mse
        + weights.paf * paf_l1
    )
    return {
        "total": total,
        "seg_bce": seg_bce,
        "heatmap_mse": heatmap_mse,
        "paf_l1": paf_l1,
    }


def compute_aux_eval_metrics(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: AuxLossWeights,
) -> dict[str, torch.Tensor]:
    losses = compute_aux_losses(output, targets, weights)
    aux_maps = output["aux_maps"]
    seg_target = targets["segmentation"]
    heatmap_target = targets["heatmap"]
    paf_target = targets["paf"]
    paf_mask = targets["paf_mask"].to(dtype=torch.float32)

    seg_logits = _resize_like(aux_maps[:, 0:1], seg_target)
    heatmap_logits = _resize_like(aux_maps[:, 1:2], heatmap_target)
    paf_pred = _resize_like(aux_maps[:, 2:4], paf_target)

    seg_pred = torch.sigmoid(seg_logits) > 0.5
    seg_truth = seg_target > 0.5
    intersection = torch.logical_and(seg_pred, seg_truth).sum(dtype=torch.float32)
    union = torch.logical_or(seg_pred, seg_truth).sum(dtype=torch.float32)
    seg_iou = intersection / union.clamp_min(1.0)

    heatmap_mae = torch.mean(torch.abs(torch.sigmoid(heatmap_logits) - heatmap_target))
    paf_masked_l1 = (torch.abs(torch.tanh(paf_pred) - paf_target) * paf_mask).sum()
    paf_masked_l1 = paf_masked_l1 / (paf_mask.sum().clamp_min(1.0) * paf_target.shape[1])

    return {
        **losses,
        "seg_iou": seg_iou,
        "heatmap_mae": heatmap_mae,
        "paf_masked_l1": paf_masked_l1,
    }


class _MetricAverager:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, int] = {}

    def update(self, metrics: dict[str, torch.Tensor], weight: int) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().item()) * weight
            self.weights[key] = self.weights.get(key, 0) + weight

    def compute(self) -> dict[str, float]:
        return {
            key: value / max(self.weights[key], 1)
            for key, value in self.sums.items()
        }


def epoch_train_aux(
    *,
    train_loader: Any,
    net: torch.nn.Module,
    optimizer: Any,
    device: torch.device,
    epoch_now: int,
    max_epoch: int,
    loss_weights: AuxLossWeights,
    clip_max_norm: float = 20.0,
    after_optimizer_step: Any | None = None,
) -> dict[str, float]:
    net.train()
    averages = _MetricAverager()
    all_len = len(train_loader)
    for i, batchdata in enumerate(train_loader):
        batch_start = time.time()
        images, targets = _prepare_aux_batch(batchdata, device)

        _, output = net(images)
        losses = compute_aux_losses(output, targets, loss_weights)
        batch_size = targets["segmentation"].shape[0]
        averages.update(losses, batch_size)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm, norm_type=2)
        optimizer.step()
        if after_optimizer_step is not None:
            after_optimizer_step(net=net, optimizer=optimizer, epoch=epoch_now, batch_index=i)

        if _dist_rank() == 0 and i % 100 == 0:
            elapsed = time.time() - batch_start
            print(
                "Epoch: {} / {} Batch: {} / {} || Aux total: {:.4f} seg: {:.4f} heatmap: {:.4f} paf: {:.4f} take {:.4f} sec.".format(
                    epoch_now - 1,
                    max_epoch,
                    i,
                    all_len,
                    losses["total"],
                    losses["seg_bce"],
                    losses["heatmap_mse"],
                    losses["paf_l1"],
                    elapsed,
                )
            )
    return averages.compute()


@torch.no_grad()
def epoch_val_aux(
    *,
    val_loader: Any,
    net: torch.nn.Module,
    device: torch.device,
    loss_weights: AuxLossWeights,
) -> dict[str, float]:
    net.eval()
    averages = _MetricAverager()
    for batchdata in val_loader:
        images, targets = _prepare_aux_batch(batchdata, device)
        _, output = net(images)
        metrics = compute_aux_eval_metrics(output, targets, loss_weights)
        averages.update(metrics, targets["segmentation"].shape[0])
    return averages.compute()
