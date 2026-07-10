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
    heatmap_focal_pos_source: str = "threshold"
    heatmap_coord: float = 0.0
    heatmap_coord_window_radius: int = 6
    heatmap_coord_temperature: float = 1.0
    heatmap_coord_huber_delta: float = 1.0
    heatmap_coord_var: float = 0.0
    heatmap_peak: float = 0.0
    heatmap_peak_center_radius: int = 1
    heatmap_peak_annulus_inner: int = 3
    heatmap_peak_annulus_outer: int = 6
    heatmap_peak_margin: float = 1.0
    heatmap_peak_temperature: float = 1.0
    heatmap_peak_min_target: float = 0.5
    heatmap_eval_peak_threshold: float = 0.5
    heatmap_eval_match_radius: float = 6.0
    paf: float = 0.25
    paf_l1: float = 1.0
    paf_angular: float = 0.0
    paf_mask_source: str = "paf"
    direction_encoding: str = "vector"


AuxLossCore = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        "torch.Tensor | None",
    ],
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
        heatmap_focal_pos_source=_choice(
            _get(train_config, "AUX_HEATMAP_FOCAL_POS_SOURCE", None),
            key="AUX_HEATMAP_FOCAL_POS_SOURCE",
            allowed={"threshold", "target_peaks"},
            default="threshold",
        ),
        heatmap_coord=float(_get(train_config, "W_AUX_HEATMAP_COORD", 0.0)),
        heatmap_coord_window_radius=int(_get(train_config, "AUX_HEATMAP_COORD_WINDOW_RADIUS", 6)),
        heatmap_coord_temperature=float(_get(train_config, "AUX_HEATMAP_COORD_TEMPERATURE", 1.0)),
        heatmap_coord_huber_delta=float(_get(train_config, "AUX_HEATMAP_COORD_HUBER_DELTA", 1.0)),
        heatmap_coord_var=float(_get(train_config, "W_AUX_HEATMAP_COORD_VAR", 0.0)),
        heatmap_peak=float(_get(train_config, "W_AUX_HEATMAP_PEAK", 0.0)),
        heatmap_peak_center_radius=int(_get(train_config, "AUX_HEATMAP_PEAK_CENTER_RADIUS", 1)),
        heatmap_peak_annulus_inner=int(_get(train_config, "AUX_HEATMAP_PEAK_ANNULUS_INNER", 3)),
        heatmap_peak_annulus_outer=int(_get(train_config, "AUX_HEATMAP_PEAK_ANNULUS_OUTER", 6)),
        heatmap_peak_margin=float(_get(train_config, "AUX_HEATMAP_PEAK_MARGIN", 1.0)),
        heatmap_peak_temperature=float(_get(train_config, "AUX_HEATMAP_PEAK_TEMPERATURE", 1.0)),
        heatmap_peak_min_target=float(_get(train_config, "AUX_HEATMAP_PEAK_MIN_TARGET", 0.5)),
        heatmap_eval_peak_threshold=float(_get(train_config, "AUX_HEATMAP_EVAL_PEAK_THRESHOLD", 0.5)),
        heatmap_eval_match_radius=float(_get(train_config, "AUX_HEATMAP_EVAL_MATCH_RADIUS", 6.0)),
        paf=float(_get(train_config, "W_AUX_PAF", 0.25)),
        paf_l1=float(_get(train_config, "W_AUX_PAF_L1", 1.0)),
        paf_angular=float(_get(train_config, "W_AUX_PAF_ANGULAR", 0.0)),
        paf_mask_source=_choice(
            _get(train_config, "AUX_PAF_MASK_SOURCE", None),
            key="AUX_PAF_MASK_SOURCE",
            allowed={"paf", "paf_and_segmentation"},
            default="paf",
        ),
        direction_encoding=_choice(
            _get(train_config, "AUX_DIRECTION_ENCODING", None),
            key="AUX_DIRECTION_ENCODING",
            allowed={"vector", "double_angle"},
            default="vector",
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
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    if positive_mask is not None:
        positive = positive_mask.bool().to(dtype=target.dtype)
    else:
        positive = (target >= pos_threshold).to(dtype=target.dtype)
    negative = 1.0 - positive
    negative_weight = (1.0 - target).clamp_min(0.0).pow(beta)

    positive_loss = -(1.0 - probabilities).pow(alpha) * probabilities.log() * positive
    negative_loss = -probabilities.pow(alpha) * (1.0 - probabilities).log() * negative_weight * negative
    if weight is not None:
        positive_loss = positive_loss * weight
        negative_loss = negative_loss * weight
        positive_count = (positive * (weight > 0.0).to(dtype=weight.dtype)).sum()
    else:
        positive_count = positive.sum()
    normalizer = positive_count.clamp_min(1.0)
    return (positive_loss.sum() + negative_loss.sum()) / normalizer


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


def _extract_target_peak_indices(heatmap_target: torch.Tensor, *, min_target: float) -> torch.Tensor:
    """Locate GT node peaks directly from the cached heatmap target.

    A peak is any pixel that equals its 3x3 max-pooled neighborhood (a local
    maximum) and is at least ``min_target``. Peaks are derived from the cached
    heatmap target's local maxima -- whatever sigma/cutoff produced the cache --
    so GT node coordinates never need to be re-plumbed through the collate
    contract; this function only ever looks at the rendered target tensor.

    Args:
        heatmap_target: ``[B,1,H,W]`` rendered node heatmap target.
        min_target: minimum target value for a local maximum to count as a peak.

    Returns:
        ``[K,3]`` long tensor of ``(batch, y, x)`` indices.
    """
    if heatmap_target.ndim != 4 or heatmap_target.shape[1] != 1:
        raise ValueError(f"heatmap_target must have shape [B,1,H,W], got {tuple(heatmap_target.shape)}")
    pooled = F.max_pool2d(heatmap_target, kernel_size=3, stride=1, padding=1)
    is_local_max = heatmap_target == pooled
    is_above_threshold = heatmap_target >= min_target
    mask = (is_local_max & is_above_threshold).squeeze(1)
    return mask.nonzero(as_tuple=False).long()


def _gather_peak_windows(
    values: torch.Tensor, peaks: torch.Tensor, *, radius: int, pad_value: float = -1e4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather (2*radius+1)^2 windows of ``values`` centered on each peak.

    ``values`` is ``[B,H,W]``. Returns ``(windows, u_y, u_x)`` where ``windows``
    is ``[K,S,S]`` (``S = 2*radius+1``) and ``u_y``/``u_x`` are the matching
    absolute (unpadded) pixel coordinates, also ``[K,S,S]``.
    """
    window_size = 2 * radius + 1
    padded = F.pad(values, (radius, radius, radius, radius), mode="constant", value=pad_value)

    k = peaks.shape[0]
    batch_idx = peaks[:, 0].long()
    y_idx = peaks[:, 1].long()
    x_idx = peaks[:, 2].long()

    offsets = torch.arange(window_size, device=values.device)
    rows = (y_idx.view(k, 1, 1) + offsets.view(1, window_size, 1)).expand(k, window_size, window_size)
    cols = (x_idx.view(k, 1, 1) + offsets.view(1, 1, window_size)).expand(k, window_size, window_size)
    batch_grid = batch_idx.view(k, 1, 1).expand(k, window_size, window_size)
    windows = padded[batch_grid, rows, cols]

    rel = offsets.float() - radius
    u_y = (y_idx.view(k, 1, 1).float() + rel.view(1, window_size, 1)).expand(k, window_size, window_size)
    u_x = (x_idx.view(k, 1, 1).float() + rel.view(1, 1, window_size)).expand(k, window_size, window_size)
    return windows, u_y, u_x


def _local_softargmax_losses(
    heatmap_logits: torch.Tensor,
    peaks: torch.Tensor,
    *,
    window_radius: int,
    temperature: float,
    huber_delta: float,
    valid_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSNT-style local soft-argmax coordinate loss and variance suppression.

    For each GT peak, a soft-argmax is computed over a ``(2*window_radius+1)``
    window of logits, and the resulting sub-pixel coordinate is compared to the
    GT peak's integer coordinate via a Huber loss. A variance term additionally
    penalizes a spread-out (non-peaky) softmax distribution within the window.
    """
    logits = heatmap_logits.float().squeeze(1)
    if peaks.shape[0] == 0:
        zero = logits.new_zeros(())
        return zero, zero

    radius = int(window_radius)
    windows, u_y, u_x = _gather_peak_windows(logits, peaks, radius=radius)
    if valid_weight is not None:
        valid_windows, _, _ = _gather_peak_windows(
            valid_weight.float().squeeze(1), peaks, radius=radius, pad_value=0.0
        )
        windows = windows.masked_fill(valid_windows <= 0.0, -1e4)
    k = peaks.shape[0]
    window_size = 2 * radius + 1

    pi = F.softmax(windows.reshape(k, -1) / temperature, dim=-1).reshape(k, window_size, window_size)

    mu_y = (pi * u_y).sum(dim=(1, 2))
    mu_x = (pi * u_x).sum(dim=(1, 2))
    mu = torch.stack((mu_y, mu_x), dim=-1)
    q = peaks[:, 1:3].float()

    coord_loss = F.huber_loss(mu, q, delta=huber_delta, reduction="none").sum(dim=-1).mean()

    diff_y = u_y - mu_y.view(k, 1, 1)
    diff_x = u_x - mu_x.view(k, 1, 1)
    sq_dist = diff_y.pow(2) + diff_x.pow(2)
    var_loss = (pi * sq_dist).sum(dim=(1, 2)).mean()

    return coord_loss, var_loss


def _peakness_margin_loss(
    heatmap_logits: torch.Tensor,
    peaks: torch.Tensor,
    *,
    center_radius: int,
    annulus_inner: int,
    annulus_outer: int,
    margin: float,
    temperature: float,
    valid_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-sum-exp margin loss between a peak's center and its surrounding annulus.

    Penalizes logits that stay high along a ridge running through a node (e.g. a
    tree trunk edge) instead of forming an isolated peak, by requiring a smooth
    max over the annulus to sit at least ``margin`` below the smooth max over the
    center region.
    """
    if not (0 <= center_radius < annulus_inner <= annulus_outer):
        raise ValueError(
            "invalid heatmap peakness geometry: require 0 <= AUX_HEATMAP_PEAK_CENTER_RADIUS < "
            "AUX_HEATMAP_PEAK_ANNULUS_INNER <= AUX_HEATMAP_PEAK_ANNULUS_OUTER, got "
            f"center_radius={center_radius}, annulus_inner={annulus_inner}, annulus_outer={annulus_outer}"
        )
    logits = heatmap_logits.float().squeeze(1)
    if peaks.shape[0] == 0:
        return logits.new_zeros(())

    radius = int(annulus_outer)
    windows, _, _ = _gather_peak_windows(logits, peaks, radius=radius)
    k = peaks.shape[0]
    window_size = 2 * radius + 1

    offsets = torch.arange(window_size, device=logits.device)
    rel = offsets.float() - radius
    dy = rel.view(window_size, 1).expand(window_size, window_size)
    dx = rel.view(1, window_size).expand(window_size, window_size)
    chebyshev = torch.maximum(dy.abs(), dx.abs())

    center_mask = (chebyshev <= center_radius).view(1, window_size, window_size).expand(k, window_size, window_size)
    annulus_mask = ((chebyshev >= annulus_inner) & (chebyshev <= annulus_outer)).view(
        1, window_size, window_size
    ).expand(k, window_size, window_size)

    valid_windows = torch.ones_like(windows, dtype=torch.bool)
    if valid_weight is not None:
        valid_values, _, _ = _gather_peak_windows(
            valid_weight.float().squeeze(1), peaks, radius=radius, pad_value=0.0
        )
        valid_windows = valid_values > 0.0
        center_mask = center_mask & valid_windows
        annulus_mask = annulus_mask & valid_windows

    center_values = windows.masked_fill(~center_mask, float("-inf")).reshape(k, -1)
    annulus_values = windows.masked_fill(~annulus_mask, float("-inf")).reshape(k, -1)

    center_valid = center_mask.reshape(k, -1).any(dim=1)
    annulus_valid = annulus_mask.reshape(k, -1).any(dim=1)

    m_center = temperature * torch.logsumexp(center_values / temperature, dim=-1)
    m_annulus = temperature * torch.logsumexp(annulus_values / temperature, dim=-1)
    valid = center_valid & annulus_valid
    loss = F.softplus(m_annulus + margin - m_center)
    return (loss * valid.to(dtype=loss.dtype)).sum() / valid.sum().clamp_min(1).to(dtype=loss.dtype)


def _paf_loss_mask(paf_mask: torch.Tensor, seg_target: torch.Tensor, weights: AuxLossWeights) -> torch.Tensor:
    if weights.paf_mask_source == "paf":
        return paf_mask
    if weights.paf_mask_source != "paf_and_segmentation":
        raise ValueError(f"unsupported paf_mask_source: {weights.paf_mask_source!r}")

    seg_mask = (seg_target > 0.5).to(dtype=paf_mask.dtype)
    if seg_mask.shape[-2:] != paf_mask.shape[-2:]:
        seg_mask = _resize_like(seg_mask, paf_mask, mode="nearest")
    return paf_mask * seg_mask


def _direction_angular_error_degrees(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    encoding: str,
) -> torch.Tensor:
    prediction = torch.tanh(prediction)
    prediction = prediction / torch.linalg.vector_norm(prediction, dim=1, keepdim=True).clamp_min(1e-6)
    target = target / torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(1e-6)
    cosine = (prediction * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    if encoding == "double_angle":
        angle = angle * 0.5
    return _masked_mean(torch.rad2deg(angle), valid_mask)


def _direction_angular_loss(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    prediction = torch.tanh(prediction)
    prediction = prediction / torch.linalg.vector_norm(prediction, dim=1, keepdim=True).clamp_min(1e-6)
    target = target / torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(1e-6)
    cosine = (prediction * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    return _masked_mean(1.0 - cosine, valid_mask)


def _compute_aux_loss_terms(
    aux_maps: torch.Tensor,
    heatmap_logits: torch.Tensor,
    seg_target: torch.Tensor,
    heatmap_target: torch.Tensor,
    paf_target: torch.Tensor,
    paf_mask: torch.Tensor,
    weights: AuxLossWeights,
    detail_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    seg_logits = _resize_like(aux_maps[:, 0:1], seg_target)
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

    needs_peaks = (
        weights.heatmap_coord > 0.0
        or weights.heatmap_coord_var > 0.0
        or weights.heatmap_peak > 0.0
        or (weights.heatmap_focal > 0.0 and weights.heatmap_focal_pos_source == "target_peaks")
    )
    peaks = _extract_target_peak_indices(heatmap_target, min_target=weights.heatmap_peak_min_target) if needs_peaks else None
    if peaks is not None:
        if heatmap_weight is not None and peaks.shape[0] > 0:
            peak_valid = heatmap_weight[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]] > 0.0
            peaks = peaks[peak_valid]

    positive_mask = None
    if weights.heatmap_focal > 0.0 and weights.heatmap_focal_pos_source == "target_peaks":
        positive_mask = torch.zeros_like(heatmap_target, dtype=torch.bool)
        if peaks.shape[0] > 0:
            positive_mask[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]] = True

    heatmap_focal = centernet_heatmap_focal_loss_with_logits(
        heatmap_logits,
        heatmap_target,
        alpha=weights.heatmap_focal_alpha,
        beta=weights.heatmap_focal_beta,
        pos_threshold=weights.heatmap_focal_pos_threshold,
        weight=heatmap_weight,
        positive_mask=positive_mask,
    )
    heatmap_probabilities = torch.sigmoid(heatmap_logits)
    ridge_mask = (heatmap_target < weights.heatmap_ridge_threshold).to(dtype=heatmap_target.dtype)
    ridge_weight = ridge_mask if heatmap_weight is None else ridge_mask * heatmap_weight
    heatmap_ridge = _weighted_mean(heatmap_probabilities.pow(2), ridge_weight)

    if weights.heatmap_coord > 0.0 or weights.heatmap_coord_var > 0.0:
        heatmap_coord, heatmap_coord_var = _local_softargmax_losses(
            heatmap_logits,
            peaks,
            window_radius=weights.heatmap_coord_window_radius,
            temperature=weights.heatmap_coord_temperature,
            huber_delta=weights.heatmap_coord_huber_delta,
            valid_weight=heatmap_weight,
        )
    else:
        heatmap_coord = zero
        heatmap_coord_var = zero

    if weights.heatmap_peak > 0.0:
        heatmap_peak = _peakness_margin_loss(
            heatmap_logits,
            peaks,
            center_radius=weights.heatmap_peak_center_radius,
            annulus_inner=weights.heatmap_peak_annulus_inner,
            annulus_outer=weights.heatmap_peak_annulus_outer,
            margin=weights.heatmap_peak_margin,
            temperature=weights.heatmap_peak_temperature,
            valid_weight=heatmap_weight,
        )
    else:
        heatmap_peak = zero

    heatmap_total = (
        weights.heatmap_mse * heatmap_mse
        + weights.heatmap_focal * heatmap_focal
        + weights.heatmap_ridge * heatmap_ridge
        + weights.heatmap_coord * heatmap_coord
        + weights.heatmap_coord_var * heatmap_coord_var
        + weights.heatmap_peak * heatmap_peak
    )

    active_paf_mask = _paf_loss_mask(paf_mask, seg_target, weights)
    paf_l1 = (torch.abs(torch.tanh(paf_pred) - paf_target) * active_paf_mask).sum()
    paf_l1 = paf_l1 / (active_paf_mask.sum().clamp_min(1.0) * paf_target.shape[1])
    paf_angular = _direction_angular_loss(paf_pred, paf_target, active_paf_mask)
    paf_total = weights.paf_l1 * paf_l1 + weights.paf_angular * paf_angular

    total = (
        weights.segmentation * seg_total
        + weights.detail * detail_total
        + weights.heatmap * heatmap_total
        + weights.paf * paf_total
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
        "heatmap_coord": heatmap_coord,
        "heatmap_coord_var": heatmap_coord_var,
        "heatmap_peak": heatmap_peak,
        "paf_total": paf_total,
        "paf_l1": paf_l1,
        "paf_angular": paf_angular,
    }


def make_aux_loss_core(weights: AuxLossWeights) -> AuxLossCore:
    def _core(
        aux_maps: torch.Tensor,
        heatmap_logits: torch.Tensor,
        seg_target: torch.Tensor,
        heatmap_target: torch.Tensor,
        paf_target: torch.Tensor,
        paf_mask: torch.Tensor,
        detail_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return _compute_aux_loss_terms(
            aux_maps, heatmap_logits, seg_target, heatmap_target, paf_target, paf_mask, weights, detail_target
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


def _select_heatmap_logits(
    output: dict[str, torch.Tensor],
    aux_maps: torch.Tensor,
    seg_target: torch.Tensor,
    heatmap_target: torch.Tensor,
) -> torch.Tensor:
    """Select direct native logits when the target is native-grid sized.

    Legacy checkpoints only expose ``aux_maps`` at image resolution, which is
    valid only for image-resolution heatmap targets.  A smaller target is an
    explicit stride-native contract and must be paired with the decoder's
    ``aux_heatmap_native`` output rather than silently supervising an
    interpolated legacy map.
    """

    native_logits = output.get("aux_heatmap_native")
    target_size = tuple(int(item) for item in heatmap_target.shape[-2:])
    seg_size = tuple(int(item) for item in seg_target.shape[-2:])
    if native_logits is None:
        if target_size != seg_size:
            raise KeyError(
                "native-grid heatmap targets require model output 'aux_heatmap_native'; "
                "set MODEL.AUX_HEAD.DECODER_STRIDE=4 and use the native aux head"
            )
        return _resize_like(aux_maps[:, 1:2], heatmap_target)
    if native_logits.ndim != 4 or native_logits.shape[1] != 1:
        raise ValueError(
            "aux_heatmap_native must be [N,1,H,W], "
            f"got shape {tuple(native_logits.shape)}"
        )
    if native_logits.shape[0] != heatmap_target.shape[0]:
        raise ValueError(
            "aux_heatmap_native batch size does not match heatmap target: "
            f"{native_logits.shape[0]} != {heatmap_target.shape[0]}"
        )
    return _resize_like(native_logits, heatmap_target)


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
    heatmap_logits = _select_heatmap_logits(output, aux_maps, seg_target, heatmap_target)

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
        return loss_core(aux_maps, heatmap_logits, seg_target, heatmap_target, paf_target, paf_mask, detail_target)
    return _compute_aux_loss_terms(
        aux_maps,
        heatmap_logits,
        seg_target,
        heatmap_target,
        paf_target,
        paf_mask,
        weights,
        detail_target,
    )


def _heatmap_peak_detection_metrics(
    heatmap_logits: torch.Tensor,
    heatmap_target: torch.Tensor,
    weights: AuxLossWeights,
    valid_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Peak-detection quality metrics: predicted local maxima vs. GT peaks.

    GT peaks come from ``_extract_target_peak_indices`` on the heatmap target;
    predicted peaks are local maxima (3x3 max-pool equality) of
    ``sigmoid(heatmap_logits)`` at or above ``weights.heatmap_eval_peak_threshold``.
    Peaks are matched per batch item within Euclidean radius
    ``weights.heatmap_eval_match_radius``, then averaged (mean) over the batch.

    Edge conventions per image (Ng = #GT peaks, Np = #pred peaks):
      - Ng == 0 and Np == 0: recall=1, precision=1, duplicate=0, background=0.
      - Ng == 0 and Np >  0: recall=1, precision=0, duplicate=0, background=Np.
      - Ng >  0 and Np == 0: recall=0, precision=1, duplicate=0, background=0.
      - otherwise: recall = fraction of GT peaks matched by >=1 pred peak;
        precision = fraction of pred peaks matched to >=1 GT peak; duplicate =
        mean over GT of max(0, #pred within radius - 1); background = #pred
        farther than radius from every GT peak.
    """
    gt_peaks = _extract_target_peak_indices(heatmap_target, min_target=weights.heatmap_peak_min_target)
    if valid_weight is not None and gt_peaks.shape[0] > 0:
        gt_valid = valid_weight[gt_peaks[:, 0], 0, gt_peaks[:, 1], gt_peaks[:, 2]] > 0.0
        gt_peaks = gt_peaks[gt_valid]
    probabilities = torch.sigmoid(heatmap_logits)
    pooled = F.max_pool2d(probabilities, kernel_size=3, stride=1, padding=1)
    pred_mask = (probabilities == pooled) & (probabilities >= weights.heatmap_eval_peak_threshold)
    if valid_weight is not None:
        pred_mask = pred_mask & (valid_weight > 0.0)
    pred_peaks = pred_mask.squeeze(1).nonzero(as_tuple=False)

    radius = float(weights.heatmap_eval_match_radius)
    recalls: list[float] = []
    precisions: list[float] = []
    duplicates: list[float] = []
    backgrounds: list[float] = []
    for batch_index in range(heatmap_target.shape[0]):
        gt_b = gt_peaks[gt_peaks[:, 0] == batch_index][:, 1:3].float()
        pred_b = pred_peaks[pred_peaks[:, 0] == batch_index][:, 1:3].float()
        n_gt = gt_b.shape[0]
        n_pred = pred_b.shape[0]
        if n_gt == 0 and n_pred == 0:
            recalls.append(1.0)
            precisions.append(1.0)
            duplicates.append(0.0)
            backgrounds.append(0.0)
        elif n_gt == 0:
            recalls.append(1.0)
            precisions.append(0.0)
            duplicates.append(0.0)
            backgrounds.append(float(n_pred))
        elif n_pred == 0:
            recalls.append(0.0)
            precisions.append(1.0)
            duplicates.append(0.0)
            backgrounds.append(0.0)
        else:
            within_radius = torch.cdist(gt_b, pred_b) <= radius
            matches_per_gt = within_radius.sum(dim=1)
            matches_per_pred = within_radius.sum(dim=0)
            recalls.append(float((matches_per_gt > 0).float().mean().item()))
            precisions.append(float((matches_per_pred > 0).float().mean().item()))
            duplicates.append(float((matches_per_gt - 1).clamp_min(0).float().mean().item()))
            backgrounds.append(float((matches_per_pred == 0).sum().item()))

    def _avg(values: list[float]) -> torch.Tensor:
        return heatmap_logits.new_tensor(sum(values) / max(len(values), 1))

    return {
        "heatmap_node_recall": _avg(recalls),
        "heatmap_node_precision": _avg(precisions),
        "heatmap_duplicate_peak_rate": _avg(duplicates),
        "heatmap_background_peaks_per_image": _avg(backgrounds),
    }


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
    heatmap_logits = _select_heatmap_logits(output, aux_maps, seg_target, heatmap_target)
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
    heatmap_peak_mask = torch.zeros_like(heatmap_target, dtype=torch.bool)
    target_peak_indices = _extract_target_peak_indices(
        heatmap_target,
        min_target=weights.heatmap_peak_min_target,
    )
    if target_peak_indices.shape[0] > 0:
        heatmap_peak_mask[
            target_peak_indices[:, 0],
            0,
            target_peak_indices[:, 1],
            target_peak_indices[:, 2],
        ] = True
    heatmap_peak_weight = heatmap_peak_mask.to(dtype=heatmap_probabilities.dtype)
    if heatmap_weight is not None:
        heatmap_peak_weight = heatmap_peak_weight * heatmap_weight
    heatmap_segmentation = _resize_like(seg_target, heatmap_target)
    heatmap_nonpeak_foreground_mask = torch.logical_and(
        heatmap_segmentation > 0.5,
        heatmap_target < weights.heatmap_ridge_threshold,
    )
    heatmap_peak_mean = _weighted_mean(heatmap_probabilities, heatmap_peak_weight)
    heatmap_nonpeak_foreground_mean = _masked_mean(heatmap_probabilities, heatmap_nonpeak_foreground_mask)
    heatmap_peak_contrast = heatmap_peak_mean - heatmap_nonpeak_foreground_mean

    heatmap_peak_detection = _heatmap_peak_detection_metrics(
        heatmap_logits, heatmap_target, weights, valid_weight=heatmap_weight
    )

    active_paf_mask = _paf_loss_mask(paf_mask, seg_target, weights)
    paf_masked_l1 = (torch.abs(torch.tanh(paf_pred) - paf_target) * active_paf_mask).sum()
    paf_masked_l1 = paf_masked_l1 / (active_paf_mask.sum().clamp_min(1.0) * paf_target.shape[1])
    direction_angular_error_deg = _direction_angular_error_degrees(
        paf_pred,
        paf_target,
        active_paf_mask,
        encoding=weights.direction_encoding,
    )
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
        **heatmap_peak_detection,
        "paf_masked_l1": paf_masked_l1,
        "direction_angular_error_deg": direction_angular_error_deg,
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
