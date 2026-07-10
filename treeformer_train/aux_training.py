from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .detail_targets import make_stdc_detail_boundary_target
from .runtime import amp_context


@dataclass(frozen=True)
class AuxLossWeights:
    segmentation: float = 1.0
    segmentation_bce: float = 1.0
    segmentation_dice: float = 0.0
    segmentation_focal: float = 0.0
    segmentation_pos_weight: float | str | None = None
    segmentation_pos_weight_max: float = 20.0
    segmentation_focal_alpha: float = 0.25
    segmentation_focal_gamma: float = 2.0
    segmentation_threshold: float = 0.5
    detail: float = 0.0
    detail_bce: float = 1.0
    detail_dice: float = 1.0
    detail_threshold: float = 0.1
    detail_scales: tuple[int, ...] = (1, 2, 4)
    detail_support_kernel_size: int = 3
    detail_eval_threshold: float = 0.5
    heatmap: float = 1.0
    heatmap_mse: float = 1.0
    heatmap_focal: float = 0.0
    heatmap_focal_alpha: float = 2.0
    heatmap_focal_beta: float = 4.0
    heatmap_focal_pos_threshold: float = 0.99
    heatmap_ridge: float = 0.0
    heatmap_ridge_threshold: float = 0.05
    heatmap_mask_source: str = "none"
    heatmap_mask_outside_weight: float = 1.0
    paf: float = 0.25
    paf_mask_source: str = "paf"


AuxLossCore = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | None"],
    dict[str, torch.Tensor],
]


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def _choice(value: Any, *, key: str, allowed: set[str], default: str) -> str:
    normalized = str(default if value is None else value).lower().replace("-", "_")
    if normalized not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return normalized


def _dist_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _mark_compile_step_begin(device: torch.device) -> None:
    if device.type != "cuda" or not hasattr(torch, "compiler"):
        return
    mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step is not None:
        mark_step()


def build_aux_loss_weights(train_config: Any) -> AuxLossWeights:
    segmentation = float(_get(train_config, "W_AUX_SEG", 1.0))
    return AuxLossWeights(
        segmentation=segmentation,
        segmentation_bce=float(_get(train_config, "W_AUX_SEG_BCE", 1.0)),
        segmentation_dice=float(_get(train_config, "W_AUX_SEG_DICE", 0.0)),
        segmentation_focal=float(_get(train_config, "W_AUX_SEG_FOCAL", 0.0)),
        segmentation_pos_weight=_get(train_config, "AUX_SEG_POS_WEIGHT", None),
        segmentation_pos_weight_max=float(_get(train_config, "AUX_SEG_POS_WEIGHT_MAX", 20.0)),
        segmentation_focal_alpha=float(_get(train_config, "AUX_SEG_FOCAL_ALPHA", 0.25)),
        segmentation_focal_gamma=float(_get(train_config, "AUX_SEG_FOCAL_GAMMA", 2.0)),
        segmentation_threshold=float(_get(train_config, "AUX_SEG_THRESHOLD", 0.5)),
        detail=float(_get(train_config, "W_AUX_DETAIL", 0.0)),
        detail_bce=float(_get(train_config, "W_AUX_DETAIL_BCE", 1.0)),
        detail_dice=float(_get(train_config, "W_AUX_DETAIL_DICE", 1.0)),
        detail_threshold=float(_get(train_config, "AUX_DETAIL_THRESHOLD", 0.1)),
        detail_scales=_as_int_tuple(_get(train_config, "AUX_DETAIL_SCALES", None), (1, 2, 4)),
        detail_support_kernel_size=int(_get(train_config, "AUX_DETAIL_SUPPORT_KERNEL_SIZE", 3)),
        detail_eval_threshold=float(_get(train_config, "AUX_DETAIL_EVAL_THRESHOLD", 0.5)),
        heatmap=float(_get(train_config, "W_AUX_HEATMAP", 1.0)),
        heatmap_mse=float(_get(train_config, "W_AUX_HEATMAP_MSE", 1.0)),
        heatmap_focal=float(_get(train_config, "W_AUX_HEATMAP_FOCAL", 0.0)),
        heatmap_focal_alpha=float(_get(train_config, "AUX_HEATMAP_FOCAL_ALPHA", 2.0)),
        heatmap_focal_beta=float(_get(train_config, "AUX_HEATMAP_FOCAL_BETA", 4.0)),
        heatmap_focal_pos_threshold=float(_get(train_config, "AUX_HEATMAP_FOCAL_POS_THRESHOLD", 0.99)),
        heatmap_ridge=float(_get(train_config, "W_AUX_HEATMAP_RIDGE", 0.0)),
        heatmap_ridge_threshold=float(_get(train_config, "AUX_HEATMAP_RIDGE_THRESHOLD", 0.05)),
        heatmap_mask_source=_choice(
            _get(train_config, "AUX_HEATMAP_MASK_SOURCE", None),
            key="AUX_HEATMAP_MASK_SOURCE",
            allowed={"none", "segmentation"},
            default="none",
        ),
        heatmap_mask_outside_weight=max(0.0, float(_get(train_config, "AUX_HEATMAP_MASK_OUTSIDE_WEIGHT", 1.0))),
        paf=float(_get(train_config, "W_AUX_PAF", 0.25)),
        paf_mask_source=_choice(
            _get(train_config, "AUX_PAF_MASK_SOURCE", None),
            key="AUX_PAF_MASK_SOURCE",
            allowed={"paf", "paf_and_segmentation"},
            default="paf",
        ),
    )


def _maybe_stack_images(images: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    if not images:
        return images
    first_shape = tuple(images[0].shape)
    if all(tuple(image.shape) == first_shape for image in images):
        return torch.stack(images, dim=0).contiguous()
    return images


def _prepare_aux_batch(
    batchdata: Any, device: torch.device
) -> tuple[torch.Tensor | list[torch.Tensor], dict[str, torch.Tensor]]:
    batch = batchdata[0]
    non_blocking = device.type == "cuda"
    images = [img.to(device, dtype=torch.float32, non_blocking=non_blocking) for img in batch[0]]
    targets = {
        "paf": batch[3].to(device, dtype=torch.float32, non_blocking=non_blocking),
        "paf_mask": batch[4].to(device, dtype=torch.bool, non_blocking=non_blocking),
        "segmentation": batch[5].to(device, dtype=torch.float32, non_blocking=non_blocking),
        "heatmap": batch[6].to(device, dtype=torch.float32, non_blocking=non_blocking),
    }
    return _maybe_stack_images(images), targets


def _resize_like(source: torch.Tensor, target: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    if mode == "nearest":
        return F.interpolate(source, size=target.shape[-2:], mode=mode)
    return F.interpolate(source, size=target.shape[-2:], mode=mode, align_corners=False)


def _prepare_binary_segmentation_target(target: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(target):
        target = target.float()
    if not torch.isfinite(target).all():
        raise ValueError("segmentation target must contain only finite values")
    min_value = float(target.detach().amin().item())
    max_value = float(target.detach().amax().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError(
            "segmentation target must be normalized to [0, 1] before loss computation; "
            f"got min={min_value:.6g}, max={max_value:.6g}"
        )
    return (target > 0.5).to(dtype=target.dtype)


def _segmentation_pos_weight(target: torch.Tensor, weights: AuxLossWeights) -> torch.Tensor | None:
    configured = weights.segmentation_pos_weight
    if configured is None:
        return None
    if isinstance(configured, str):
        if configured.lower() != "auto":
            raise ValueError(f"unsupported AUX_SEG_POS_WEIGHT value: {configured!r}")
        binary_target = (target > 0.5).to(dtype=target.dtype)
        positives = binary_target.sum().clamp_min(1.0)
        negatives = (binary_target.numel() - binary_target.sum()).clamp_min(1.0)
        value = (negatives / positives).clamp(max=weights.segmentation_pos_weight_max)
    else:
        value = torch.as_tensor(float(configured), dtype=target.dtype, device=target.device).clamp_min(0.0)
    return value.reshape(1)


def binary_dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def centernet_heatmap_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    pos_threshold: float = 0.99,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positive = (target >= pos_threshold).to(dtype=target.dtype)
    negative = 1.0 - positive
    negative_weight = (1.0 - target).clamp_min(0.0).pow(beta)

    positive_loss = -(1.0 - probabilities).pow(alpha) * probabilities.log() * positive
    negative_loss = -probabilities.pow(alpha) * (1.0 - probabilities).log() * negative_weight * negative
    if weight is not None:
        positive_loss = positive_loss * weight
        negative_loss = negative_loss * weight
        positive_count = (positive * weight).sum()
        negative_count = (negative * weight).sum()
    else:
        positive_count = positive.sum()
        negative_count = negative.sum()
    normalizer = torch.where(positive_count > 0.0, positive_count, negative_count.clamp_min(1.0))
    return (positive_loss.sum() + negative_loss.sum()) / normalizer.clamp_min(1.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _heatmap_loss_weight(
    seg_target: torch.Tensor, heatmap_target: torch.Tensor, weights: AuxLossWeights
) -> torch.Tensor | None:
    if weights.heatmap_mask_source == "none":
        return None
    if weights.heatmap_mask_source != "segmentation":
        raise ValueError(f"unsupported heatmap_mask_source: {weights.heatmap_mask_source!r}")

    seg_mask = (seg_target > 0.5).to(dtype=heatmap_target.dtype)
    if seg_mask.shape[-2:] != heatmap_target.shape[-2:]:
        seg_mask = _resize_like(seg_mask, heatmap_target, mode="nearest")
    outside_weight = float(weights.heatmap_mask_outside_weight)
    if outside_weight <= 0.0:
        return seg_mask
    return seg_mask + (1.0 - seg_mask) * outside_weight


def _paf_loss_mask(paf_mask: torch.Tensor, seg_target: torch.Tensor, weights: AuxLossWeights) -> torch.Tensor:
    if weights.paf_mask_source == "paf":
        return paf_mask
    if weights.paf_mask_source != "paf_and_segmentation":
        raise ValueError(f"unsupported paf_mask_source: {weights.paf_mask_source!r}")

    seg_mask = (seg_target > 0.5).to(dtype=paf_mask.dtype)
    if seg_mask.shape[-2:] != paf_mask.shape[-2:]:
        seg_mask = _resize_like(seg_mask, paf_mask, mode="nearest")
    return paf_mask * seg_mask


def _compute_aux_loss_terms(
    aux_maps: torch.Tensor,
    seg_target: torch.Tensor,
    heatmap_target: torch.Tensor,
    paf_target: torch.Tensor,
    paf_mask: torch.Tensor,
    weights: AuxLossWeights,
    detail_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    seg_logits = _resize_like(aux_maps[:, 0:1], seg_target)
    heatmap_logits = _resize_like(aux_maps[:, 1:2], heatmap_target)
    paf_pred = _resize_like(aux_maps[:, 2:4], paf_target)

    seg_pos_weight = _segmentation_pos_weight(seg_target, weights)
    seg_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_target, pos_weight=seg_pos_weight)
    seg_dice = binary_dice_loss_with_logits(seg_logits, seg_target)
    seg_focal = binary_focal_loss_with_logits(
        seg_logits,
        seg_target,
        alpha=weights.segmentation_focal_alpha,
        gamma=weights.segmentation_focal_gamma,
    )
    seg_total = (
        weights.segmentation_bce * seg_bce
        + weights.segmentation_dice * seg_dice
        + weights.segmentation_focal * seg_focal
    )
    zero = seg_total.new_zeros(())
    detail_bce = zero
    detail_dice = zero
    detail_total = zero
    if aux_maps.shape[1] >= 5:
        if detail_target is None:
            raise ValueError(
                "detail_target must be provided when aux_maps has a 5th (detail boundary) "
                "channel active. Compute it once from the segmentation target at the call "
                "site (see compute_aux_losses) and pass it in; this function does not "
                "recompute it implicitly."
            )
        detail_logits = _resize_like(aux_maps[:, 4:5], detail_target)
        detail_bce = F.binary_cross_entropy_with_logits(detail_logits, detail_target)
        detail_dice = binary_dice_loss_with_logits(detail_logits, detail_target)
        detail_total = weights.detail_bce * detail_bce + weights.detail_dice * detail_dice
    heatmap_error = (torch.sigmoid(heatmap_logits) - heatmap_target).pow(2)
    heatmap_weight = _heatmap_loss_weight(seg_target, heatmap_target, weights)
    heatmap_mse = heatmap_error.mean() if heatmap_weight is None else _weighted_mean(heatmap_error, heatmap_weight)
    heatmap_focal = centernet_heatmap_focal_loss_with_logits(
        heatmap_logits,
        heatmap_target,
        alpha=weights.heatmap_focal_alpha,
        beta=weights.heatmap_focal_beta,
        pos_threshold=weights.heatmap_focal_pos_threshold,
        weight=heatmap_weight,
    )
    heatmap_probabilities = torch.sigmoid(heatmap_logits)
    ridge_mask = (heatmap_target < weights.heatmap_ridge_threshold).to(dtype=heatmap_target.dtype)
    ridge_weight = ridge_mask if heatmap_weight is None else ridge_mask * heatmap_weight
    heatmap_ridge = _weighted_mean(heatmap_probabilities.pow(2), ridge_weight)
    heatmap_total = (
        weights.heatmap_mse * heatmap_mse
        + weights.heatmap_focal * heatmap_focal
        + weights.heatmap_ridge * heatmap_ridge
    )

    active_paf_mask = _paf_loss_mask(paf_mask, seg_target, weights)
    paf_l1 = (torch.abs(torch.tanh(paf_pred) - paf_target) * active_paf_mask).sum()
    paf_l1 = paf_l1 / (active_paf_mask.sum().clamp_min(1.0) * paf_target.shape[1])

    total = (
        weights.segmentation * seg_total
        + weights.detail * detail_total
        + weights.heatmap * heatmap_total
        + weights.paf * paf_l1
    )
    # Deliberately scalar-only: _MetricAverager.update() calls
    # float(value.detach().item()) on every value in this dict (directly, or via
    # compute_aux_eval_metrics' losses dict below). Do not add non-scalar tensors
    # (e.g. detail_target) here -- see compute_aux_eval_metrics for how the
    # single computed detail_target is threaded through and reused without ever
    # entering a metrics dict.
    return {
        "total": total,
        "seg_total": seg_total,
        "seg_bce": seg_bce,
        "seg_dice": seg_dice,
        "seg_focal": seg_focal,
        "detail_total": detail_total,
        "detail_bce": detail_bce,
        "detail_dice": detail_dice,
        "heatmap_total": heatmap_total,
        "heatmap_mse": heatmap_mse,
        "heatmap_focal": heatmap_focal,
        "heatmap_ridge": heatmap_ridge,
        "paf_l1": paf_l1,
    }


def make_aux_loss_core(weights: AuxLossWeights) -> AuxLossCore:
    def _core(
        aux_maps: torch.Tensor,
        seg_target: torch.Tensor,
        heatmap_target: torch.Tensor,
        paf_target: torch.Tensor,
        paf_mask: torch.Tensor,
        detail_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return _compute_aux_loss_terms(
            aux_maps, seg_target, heatmap_target, paf_target, paf_mask, weights, detail_target
        )

    return _core


@dataclass
class AuxLossComputer:
    weights: AuxLossWeights
    core: AuxLossCore | None = None

    def __post_init__(self) -> None:
        if self.core is None:
            self.core = make_aux_loss_core(self.weights)

    def __call__(
        self,
        output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        detail_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return compute_aux_losses(output, targets, self.weights, loss_core=self.core, detail_target=detail_target)


def build_aux_loss_computer(
    weights: AuxLossWeights,
    *,
    compile_core: bool = False,
    compile_options: dict[str, Any] | None = None,
) -> AuxLossComputer:
    core = make_aux_loss_core(weights)
    if compile_core:
        core = torch.compile(core, **(compile_options or {}))
    return AuxLossComputer(weights=weights, core=core)


def compute_aux_losses(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: AuxLossWeights,
    *,
    loss_core: AuxLossCore | None = None,
    detail_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    aux_maps = output.get("aux_maps")
    if aux_maps is None:
        raise KeyError("model output must contain 'aux_maps' for aux supervised training")
    if aux_maps.shape[1] < 4:
        raise ValueError(f"aux_maps must have at least 4 channels, got shape {tuple(aux_maps.shape)}")
    if weights.detail > 0.0 and aux_maps.shape[1] < 5:
        raise ValueError("detail boundary loss requires MODEL.AUX_HEAD.OUT_CHANNELS>=5")

    seg_target = _prepare_binary_segmentation_target(targets["segmentation"])
    heatmap_target = targets["heatmap"]
    paf_target = targets["paf"]
    paf_mask = targets["paf_mask"].to(dtype=torch.float32)

    # Build the detail target only when the caller has not already shared one.
    # compute_aux_eval_metrics creates it first so its loss and metric paths reuse
    # a single tensor. _compute_aux_loss_terms itself always requires an explicit
    # value and refuses to recompute it.
    if aux_maps.shape[1] >= 5 and detail_target is None:
        detail_target = make_stdc_detail_boundary_target(
            seg_target,
            threshold=weights.detail_threshold,
            scales=weights.detail_scales,
            support_kernel_size=weights.detail_support_kernel_size,
        )

    if loss_core is not None:
        return loss_core(aux_maps, seg_target, heatmap_target, paf_target, paf_mask, detail_target)
    return _compute_aux_loss_terms(aux_maps, seg_target, heatmap_target, paf_target, paf_mask, weights, detail_target)


def compute_aux_eval_metrics(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: AuxLossWeights,
    *,
    loss_core: AuxLossCore | None = None,
    detail_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    aux_maps = output["aux_maps"]
    seg_target = _prepare_binary_segmentation_target(targets["segmentation"])

    # Compute detail_target exactly once here (or use the caller-supplied one)
    # and pass it into compute_aux_losses so it does not compute its own copy.
    # The result is reused below for this function's own detail metrics instead
    # of being read back out of the losses dict, which must stay scalar-only.
    if aux_maps.shape[1] >= 5 and detail_target is None:
        detail_target = make_stdc_detail_boundary_target(
            seg_target,
            threshold=weights.detail_threshold,
            scales=weights.detail_scales,
            support_kernel_size=weights.detail_support_kernel_size,
        )

    losses = compute_aux_losses(output, targets, weights, loss_core=loss_core, detail_target=detail_target)
    heatmap_target = targets["heatmap"]
    paf_target = targets["paf"]
    paf_mask = targets["paf_mask"].to(dtype=torch.float32)

    seg_logits = _resize_like(aux_maps[:, 0:1], seg_target)
    heatmap_logits = _resize_like(aux_maps[:, 1:2], heatmap_target)
    paf_pred = _resize_like(aux_maps[:, 2:4], paf_target)

    seg_probabilities = torch.sigmoid(seg_logits)
    seg_pred = seg_probabilities > weights.segmentation_threshold
    seg_truth = seg_target > 0.5
    intersection = torch.logical_and(seg_pred, seg_truth).sum(dtype=torch.float32)
    union = torch.logical_or(seg_pred, seg_truth).sum(dtype=torch.float32)
    seg_iou = intersection / union.clamp_min(1.0)
    pred_positive = seg_pred.sum(dtype=torch.float32)
    truth_positive = seg_truth.sum(dtype=torch.float32)
    seg_precision = intersection / pred_positive.clamp_min(1.0)
    seg_recall = intersection / truth_positive.clamp_min(1.0)
    seg_dice_score = (2.0 * intersection) / (pred_positive + truth_positive).clamp_min(1.0)
    seg_soft_dice_score = 1.0 - losses["seg_dice"]
    pred_positive_rate = pred_positive / float(seg_pred.numel())
    target_positive_rate = truth_positive / float(seg_truth.numel())

    heatmap_abs_error = torch.abs(torch.sigmoid(heatmap_logits) - heatmap_target)
    heatmap_mae = torch.mean(heatmap_abs_error)
    heatmap_weight = _heatmap_loss_weight(seg_target, heatmap_target, weights)
    masked_heatmap_mae = heatmap_mae if heatmap_weight is None else _weighted_mean(heatmap_abs_error, heatmap_weight)
    heatmap_probabilities = torch.sigmoid(heatmap_logits)
    heatmap_peak_mask = heatmap_target >= weights.heatmap_focal_pos_threshold
    heatmap_nonpeak_foreground_mask = torch.logical_and(
        seg_target > 0.5,
        heatmap_target < weights.heatmap_ridge_threshold,
    )
    heatmap_peak_mean = _masked_mean(heatmap_probabilities, heatmap_peak_mask)
    heatmap_nonpeak_foreground_mean = _masked_mean(heatmap_probabilities, heatmap_nonpeak_foreground_mask)
    heatmap_peak_contrast = heatmap_peak_mean - heatmap_nonpeak_foreground_mean

    active_paf_mask = _paf_loss_mask(paf_mask, seg_target, weights)
    paf_masked_l1 = (torch.abs(torch.tanh(paf_pred) - paf_target) * active_paf_mask).sum()
    paf_masked_l1 = paf_masked_l1 / (active_paf_mask.sum().clamp_min(1.0) * paf_target.shape[1])
    if aux_maps.shape[1] >= 5:
        # detail_target was computed once above (or supplied by the caller) and
        # passed into compute_aux_losses; reuse that same local variable here
        # instead of recomputing it or reading it back out of the losses dict.
        if detail_target is None:
            raise ValueError(
                "detail_target must have been computed above when aux_maps has a 5th "
                "(detail boundary) channel active; this indicates a logic error in the "
                "computation guard, not a reason to silently recompute it here"
            )
        detail_logits = _resize_like(aux_maps[:, 4:5], detail_target)
        detail_probabilities = torch.sigmoid(detail_logits)
        detail_pred = detail_probabilities > weights.detail_eval_threshold
        detail_truth = detail_target > 0.5
        detail_intersection = torch.logical_and(detail_pred, detail_truth).sum(dtype=torch.float32)
        detail_union = torch.logical_or(detail_pred, detail_truth).sum(dtype=torch.float32)
        detail_pred_positive = detail_pred.sum(dtype=torch.float32)
        detail_truth_positive = detail_truth.sum(dtype=torch.float32)
        detail_iou = detail_intersection / detail_union.clamp_min(1.0)
        detail_dice_score = (2.0 * detail_intersection) / (detail_pred_positive + detail_truth_positive).clamp_min(1.0)
        detail_soft_dice_score = 1.0 - losses["detail_dice"]
        detail_pred_positive_rate = detail_pred_positive / float(detail_pred.numel())
        detail_target_positive_rate = detail_truth_positive / float(detail_truth.numel())
    else:
        detail_iou = seg_iou.new_zeros(())
        detail_dice_score = seg_iou.new_zeros(())
        detail_soft_dice_score = seg_iou.new_zeros(())
        detail_pred_positive_rate = seg_iou.new_zeros(())
        detail_target_positive_rate = seg_iou.new_zeros(())

    return {
        **losses,
        "seg_iou": seg_iou,
        "seg_dice_score": seg_dice_score,
        "seg_soft_dice_score": seg_soft_dice_score,
        "seg_precision": seg_precision,
        "seg_recall": seg_recall,
        "pred_positive_rate": pred_positive_rate,
        "target_positive_rate": target_positive_rate,
        "detail_iou": detail_iou,
        "detail_dice_score": detail_dice_score,
        "detail_soft_dice_score": detail_soft_dice_score,
        "detail_pred_positive_rate": detail_pred_positive_rate,
        "detail_target_positive_rate": detail_target_positive_rate,
        "heatmap_mae": heatmap_mae,
        "masked_heatmap_mae": masked_heatmap_mae,
        "heatmap_peak_mean": heatmap_peak_mean,
        "heatmap_nonpeak_foreground_mean": heatmap_nonpeak_foreground_mean,
        "heatmap_peak_contrast": heatmap_peak_contrast,
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
        return {key: value / max(self.weights[key], 1) for key, value in self.sums.items()}


def epoch_train_aux(
    *,
    train_loader: Any,
    net: torch.nn.Module,
    optimizer: Any,
    device: torch.device,
    epoch_now: int,
    max_epoch: int,
    loss_weights: AuxLossWeights,
    loss_computer: AuxLossComputer | None = None,
    clip_max_norm: float = 20.0,
    after_optimizer_step: Any | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    net.train()
    averages = _MetricAverager()
    all_len = len(train_loader)
    if loss_computer is None:
        loss_computer = AuxLossComputer(loss_weights)
    for i, batchdata in enumerate(train_loader):
        batch_start = time.time()
        images, targets = _prepare_aux_batch(batchdata, device)

        _mark_compile_step_begin(device)
        with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            _, output = net(images)
            losses = loss_computer(output, targets)
        batch_size = targets["segmentation"].shape[0]
        averages.update(losses, batch_size)

        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is None:
            losses["total"].backward()
        else:
            grad_scaler.scale(losses["total"]).backward()
            grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm, norm_type=2)
        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
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
                    losses["seg_total"],
                    losses["heatmap_total"],
                    losses["paf_l1"],
                    elapsed,
                )
            )
    return averages.compute()


@torch.inference_mode()
def epoch_val_aux(
    *,
    val_loader: Any,
    net: torch.nn.Module,
    device: torch.device,
    loss_weights: AuxLossWeights,
    loss_computer: AuxLossComputer | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    net.eval()
    averages = _MetricAverager()
    if loss_computer is None:
        loss_computer = AuxLossComputer(loss_weights)
    for batchdata in val_loader:
        images, targets = _prepare_aux_batch(batchdata, device)
        _mark_compile_step_begin(device)
        with amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
            _, output = net(images)
        metrics = compute_aux_eval_metrics(output, targets, loss_weights, loss_core=loss_computer.core)
        averages.update(metrics, targets["segmentation"].shape[0])
    return averages.compute()
