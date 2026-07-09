#!/usr/bin/env python3
"""Render TreeFormer aux-map inference panels for qualitative validation.

The aux head emits segmentation, node heatmap and PAF direction maps.  This
script compares those predictions against the legacy dataloader targets and
also renders a STDC-style detail edge map derived from the segmentation mask.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infer_panel_treeformer import (
    add_label,
    load_checkpoint_mapping,
    load_config_for_run,
    make_panel_grid,
    select_state_dict,
)


def _resampling_bilinear() -> int:
    return getattr(Image, "Resampling", Image).BILINEAR


def image_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a legacy normalized CHW image tensor to RGB PIL."""

    array = tensor.detach().cpu().float().clamp(-1.0, 1.0).add(1.0).mul(127.5)
    array = array.byte().permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return Image.fromarray(array[:, :, :3], mode="RGB")


def normalize01(values: torch.Tensor | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    array = values.detach().cpu().float().numpy() if isinstance(values, torch.Tensor) else np.asarray(values, dtype=np.float32)
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(array.min()) if array.size else 0.0
    max_value = float(array.max()) if array.size else 0.0
    if max_value - min_value < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def grayscale_to_pil(values: torch.Tensor | np.ndarray, *, normalize: bool = False) -> Image.Image:
    array = normalize01(values) if normalize else np.asarray(values.detach().cpu() if isinstance(values, torch.Tensor) else values, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="L").convert("RGB")


def heatmap_to_pil(values: torch.Tensor | np.ndarray) -> Image.Image:
    """Small dependency-free blue-green-yellow-red heatmap."""

    array = np.clip(normalize01(values), 0.0, 1.0)
    red = np.clip(1.5 * array - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(array - 0.55) * 2.2, 0.0, 1.0)
    blue = np.clip(1.2 - 2.0 * array, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def overlay_map(image: Image.Image, mask: torch.Tensor | np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    values = np.asarray(mask.detach().cpu() if isinstance(mask, torch.Tensor) else mask, dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    if values.ndim == 3:
        values = values.squeeze()
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    mixed = base * (1.0 - values[..., None] * alpha) + color_array * (values[..., None] * alpha)
    return Image.fromarray(np.clip(mixed, 0.0, 255.0).astype(np.uint8), mode="RGB")


def paf_to_rgb(paf: torch.Tensor | np.ndarray) -> Image.Image:
    """Visualize a two-channel vector field as direction color and magnitude value."""

    array = paf.detach().cpu().float().numpy() if isinstance(paf, torch.Tensor) else np.asarray(paf, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 2:
        x = array[0]
        y = array[1]
    elif array.ndim == 3 and array.shape[-1] == 2:
        x = array[..., 0]
        y = array[..., 1]
    else:
        raise ValueError(f"PAF must be shaped [2,H,W] or [H,W,2], got {array.shape}")

    magnitude = np.clip(np.sqrt(x * x + y * y), 0.0, 1.0)
    angle = (np.arctan2(y, x) + np.pi) / (2.0 * np.pi)
    red = np.clip(np.abs(angle * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(angle * 6.0 - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 - np.abs(angle * 6.0 - 4.0), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1) * magnitude[..., None]
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def make_detail_edge_map(mask: torch.Tensor, *, threshold: float = 0.1, scales: tuple[int, ...] = (1, 2, 4)) -> torch.Tensor:
    """Create an STDC-style multi-scale detail edge map from a dense mask."""

    mask = mask.detach().float()
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [H,W], [1,H,W] or [N,1,H,W], got {tuple(mask.shape)}")

    kernel = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        device=mask.device,
        dtype=mask.dtype,
    ).view(1, 1, 3, 3)
    height, width = mask.shape[-2:]
    edges = []
    for scale in scales:
        if scale <= 1:
            scaled = mask
        else:
            scaled = F.interpolate(mask, scale_factor=1.0 / float(scale), mode="bilinear", align_corners=False)
        edge = F.conv2d(scaled, kernel, padding=1).abs()
        edge = (edge > threshold).to(dtype=mask.dtype)
        if edge.shape[-2:] != (height, width):
            edge = F.interpolate(edge, size=(height, width), mode="nearest")
        edges.append(edge)
    combined = torch.stack(edges, dim=0).amax(dim=0)
    fine_support = F.max_pool2d(edges[0], kernel_size=5, stride=1, padding=2)
    combined = combined * fine_support
    return combined.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def resize_aux_maps(aux_maps: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if aux_maps.ndim != 4:
        raise ValueError(f"aux_maps must be [N,C,H,W], got {tuple(aux_maps.shape)}")
    if aux_maps.shape[-2:] == target_size:
        return aux_maps
    return F.interpolate(aux_maps, size=target_size, mode="bilinear", align_corners=False)


def load_aux_model(
    *,
    checkpoint_path: Path,
    device: torch.device,
    weights: str,
    legacy_key: str,
    strict: bool,
) -> torch.nn.Module:
    from models import build_model

    checkpoint = load_checkpoint_mapping(checkpoint_path)
    config = load_config_for_run(config_path=None, default_config_path=None, checkpoint=checkpoint)
    if not bool(getattr(config.MODEL.AUX_HEAD, "ENABLED", False)):
        raise ValueError("checkpoint config does not enable MODEL.AUX_HEAD")
    config.MODEL.GRAPH_OUTPUT_ENABLED = False

    model_args = SimpleNamespace(use_gnn=False, use_mst_train=True)
    model = build_model(config, args=model_args).to(device)
    state_dict = select_state_dict(checkpoint, weights=weights, legacy_key=legacy_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if missing:
            print(f"[WARN] {checkpoint_path.name}: missing keys={len(missing)}")
        if unexpected:
            print(f"[WARN] {checkpoint_path.name}: unexpected keys={len(unexpected)}")
    model.eval()
    return model


def predict_aux_maps(model: torch.nn.Module, image: torch.Tensor, device: torch.device, target_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        _, output = model([image.to(device)])
        aux_maps = output.get("aux_maps")
        if aux_maps is None:
            raise KeyError("model output does not contain aux_maps")
        aux_maps = resize_aux_maps(aux_maps.detach().cpu(), target_size)[0]
    return {
        "segmentation": torch.sigmoid(aux_maps[0]).clamp(0.0, 1.0),
        "heatmap": torch.sigmoid(aux_maps[1]).clamp(0.0, 1.0),
        "paf": torch.tanh(aux_maps[2:4]).clamp(-1.0, 1.0),
    }


def make_aux_panel(
    *,
    image: torch.Tensor,
    gt_segmentation: torch.Tensor,
    gt_heatmap: torch.Tensor,
    gt_paf: torch.Tensor,
    prediction: Mapping[str, torch.Tensor],
    columns: int,
    panel_width: int,
    pad: int,
) -> Image.Image:
    input_image = image_tensor_to_pil(image)
    gt_segmentation = gt_segmentation.float().clamp(0.0, 1.0)
    gt_heatmap = gt_heatmap.float().clamp(0.0, 1.0)
    gt_paf = gt_paf.float().permute(2, 0, 1).clamp(-1.0, 1.0) if gt_paf.ndim == 3 and gt_paf.shape[-1] == 2 else gt_paf.float()

    pred_segmentation = prediction["segmentation"]
    pred_heatmap = prediction["heatmap"]
    pred_paf = prediction["paf"]

    gt_edge = make_detail_edge_map(gt_segmentation)
    pred_edge = make_detail_edge_map(pred_segmentation)
    gt_paf_magnitude = torch.linalg.vector_norm(gt_paf, dim=0).clamp(0.0, 1.0)
    pred_paf_magnitude = torch.linalg.vector_norm(pred_paf, dim=0).clamp(0.0, 1.0)

    panels = [
        add_label(input_image, "Input"),
        add_label(overlay_map(input_image, gt_segmentation, (40, 210, 80)), "GT segmentation overlay"),
        add_label(overlay_map(input_image, pred_segmentation, (240, 80, 40)), "Pred segmentation overlay"),
        add_label(grayscale_to_pil(gt_edge), "GT detail edge"),
        add_label(grayscale_to_pil(pred_edge), "Pred detail edge"),
        add_label(heatmap_to_pil(gt_heatmap), "GT node heatmap"),
        add_label(heatmap_to_pil(pred_heatmap), "Pred node heatmap"),
        add_label(grayscale_to_pil(gt_paf_magnitude), "GT PAF magnitude"),
        add_label(grayscale_to_pil(pred_paf_magnitude), "Pred PAF magnitude"),
        add_label(paf_to_rgb(gt_paf), "GT PAF direction"),
        add_label(paf_to_rgb(pred_paf), "Pred PAF direction"),
    ]
    return make_panel_grid(panels, columns=columns, pad=pad, panel_width=panel_width)


def write_aux_summary_json(path: Path, *, sample_id: str, prediction: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> None:
    pred_seg = prediction["segmentation"]
    gt_seg = targets["segmentation"]
    pred_heatmap = prediction["heatmap"]
    gt_heatmap = targets["heatmap"]
    pred_paf = prediction["paf"]
    gt_paf = targets["paf"]
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "pred_segmentation_mean": float(pred_seg.mean()),
        "gt_segmentation_mean": float(gt_seg.mean()),
        "pred_heatmap_max": float(pred_heatmap.max()),
        "gt_heatmap_max": float(gt_heatmap.max()),
        "pred_paf_magnitude_mean": float(torch.linalg.vector_norm(pred_paf, dim=0).mean()),
        "gt_paf_magnitude_mean": float(torch.linalg.vector_norm(gt_paf, dim=0).mean()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TreeFormer aux-map inference panel renderer")
    parser.add_argument("--legacy-split-root", required=True, help="Legacy TreeFormer split root containing img/ and data/")
    parser.add_argument("--checkpoint", required=True, help="Hydra aux-supervised checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Directory where aux panels are saved")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--weights", default="auto", choices=("auto", "ema", "model"), help="Checkpoint weights to load")
    parser.add_argument("--legacy-key", default="net", help="Legacy checkpoint state_dict key")
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading")
    parser.add_argument("--max-size", type=int, default=128, help="Dataset max image size")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to render")
    parser.add_argument("--columns", type=int, default=3, help="Grid columns")
    parser.add_argument("--panel-width", type=int, default=320, help="Rendered width of each panel cell")
    parser.add_argument("--pad", type=int, default=8, help="Whitespace between panel cells")
    parser.add_argument("--save-json", action="store_true", help="Save compact per-sample map statistics JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    from train_mst import LoadCNNDataset

    dataset = LoadCNNDataset(
        parent_path=args.legacy_split_root,
        max_size=int(args.max_size),
        max_change_light_rate=0.3,
        is_train=False,
        is_rotate=False,
    )
    model = load_aux_model(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        weights=str(args.weights),
        legacy_key=str(args.legacy_key),
        strict=bool(args.strict),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset) if args.limit is None or args.limit <= 0 else min(int(args.limit), len(dataset))
    print(f"Rendering {limit} aux panels from {args.legacy_split_root}")

    for index in range(limit):
        image, label, _nodes, _edges, pafs, _mask, unet, heatmap, sample_id = dataset[index]
        target_size = (int(unet.shape[-2]), int(unet.shape[-1]))
        prediction = predict_aux_maps(model, image, device, target_size)
        panel = make_aux_panel(
            image=image,
            gt_segmentation=unet,
            gt_heatmap=heatmap,
            gt_paf=pafs,
            prediction=prediction,
            columns=int(args.columns),
            panel_width=int(args.panel_width),
            pad=int(args.pad),
        )
        sample_name = Path(str(sample_id or label)).stem
        panel_path = output_dir / f"{sample_name}_aux_panel.png"
        panel.save(panel_path)
        if args.save_json:
            targets = {
                "segmentation": unet.float(),
                "heatmap": heatmap.float(),
                "paf": pafs.float().permute(2, 0, 1),
            }
            write_aux_summary_json(output_dir / f"{sample_name}_aux_summary.json", sample_id=sample_name, prediction=prediction, targets=targets)
        print(f"[{index + 1}/{limit}] saved: {panel_path}")


if __name__ == "__main__":
    main()
