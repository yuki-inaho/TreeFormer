#!/usr/bin/env python3
"""Render per-image TreeFormer inference summary panels.

The script supports current Hydra checkpoints and older TreeFormer checkpoints.
It writes one panel image per input image and can also save predicted graph JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms import functional as TVF

from treeformer_train.config import AttrDict, load_legacy_yaml, make_legacy_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class RunSpec:
    label: str
    checkpoint: Path
    mode: str
    config: Path | None = None

    @property
    def use_mst(self) -> bool:
        return self.mode in {"mst", "mst-dist", "vr-mst"}

    @property
    def use_distance(self) -> bool:
        return self.mode == "mst-dist"


@dataclass
class AuxDiagnosticContext:
    samples_by_name: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    model: torch.nn.Module | None
    device: torch.device
    show_heatmap: bool
    show_paf: bool
    loader_name: str


def parse_run_spec(spec: str) -> RunSpec:
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) == 3:
        label, checkpoint, mode = parts
        config = None
    elif len(parts) == 4:
        label, config, checkpoint, mode = parts
    else:
        raise argparse.ArgumentTypeError("--run must be 'LABEL|CHECKPOINT|MODE' or 'LABEL|CONFIG|CHECKPOINT|MODE'")
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    if mode not in {"raw", "mst", "mst-dist", "vr-mst"}:
        raise argparse.ArgumentTypeError("run mode must be one of: raw, mst, mst-dist, vr-mst")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint not found: {checkpoint_path}")
    config_path = Path(config) if config is not None else None
    if config_path is not None and not config_path.is_file():
        raise argparse.ArgumentTypeError(f"config not found: {config_path}")
    return RunSpec(label=label, checkpoint=checkpoint_path, mode=mode, config=config_path)


def discover_images(image_dir: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.iterdir()
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no image files found in: {root}")
    return images


def load_config_for_run(
    *,
    config_path: Path | None,
    default_config_path: Path | None,
    checkpoint: Mapping[str, Any],
) -> AttrDict:
    path = config_path or default_config_path
    if path is not None:
        return AttrDict(load_legacy_yaml(path))
    embedded = checkpoint.get("config")
    if isinstance(embedded, Mapping):
        return make_legacy_config(embedded)
    raise ValueError("no config was provided and checkpoint does not contain a Hydra config payload")


def select_state_dict(
    checkpoint: Mapping[str, Any],
    *,
    weights: str = "auto",
    legacy_key: str = "net",
) -> dict[str, torch.Tensor]:
    """Select model weights from Hydra or legacy checkpoint payloads.

    ``auto`` prefers EMA shadow weights because Hydra validation uses EMA when
    enabled, then falls back to current-model and legacy keys.
    """

    candidates: list[Any]
    if weights == "auto":
        ema = checkpoint.get("ema")
        ema_shadow = ema.get("shadow") if isinstance(ema, Mapping) else None
        candidates = [ema_shadow, checkpoint.get("model"), checkpoint.get(legacy_key), checkpoint.get("state_dict")]
    elif weights == "ema":
        ema = checkpoint.get("ema")
        candidates = [ema.get("shadow") if isinstance(ema, Mapping) else None]
    elif weights == "model":
        candidates = [checkpoint.get("model"), checkpoint.get(legacy_key), checkpoint.get("state_dict")]
    else:
        raise ValueError(f"unsupported weights selection: {weights!r}")

    for state_dict in candidates:
        if isinstance(state_dict, Mapping):
            return {
                str(key).removeprefix("module."): value
                for key, value in state_dict.items()
                if isinstance(value, torch.Tensor)
            }
    raise ValueError(f"checkpoint does not contain usable weights for selection {weights!r}")


def load_model(
    config: AttrDict,
    checkpoint_path: Path,
    *,
    device: torch.device,
    strict: bool,
    weights: str,
    legacy_key: str,
) -> torch.nn.Module:
    from models import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")

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


def load_checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    return checkpoint


def load_image_for_model(path: Path, max_size: int | None) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    model_image = image
    if max_size is not None and max_size > 0:
        width, height = model_image.size
        scale = min(1.0, float(max_size) / float(max(width, height)))
        if scale < 1.0:
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            model_image = model_image.resize(new_size, resample=_resampling_bilinear())
    tensor = TVF.to_tensor(model_image).to(dtype=torch.float32)
    tensor = tensor.mul(2.0).sub(1.0).clamp(-1.0, 1.0).contiguous()
    return image, tensor


def run_inference(
    *,
    model: torch.nn.Module,
    config: AttrDict,
    tensor: torch.Tensor,
    device: torch.device,
    run: RunSpec,
    nms: bool,
    distance_weight: float,
    graph_node_segmentation_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    from inference_treeformer import relation_infer

    with torch.inference_mode():
        images = [tensor.to(device, non_blocking=True)]
        h, out = model(images)
        node_valid_mask = graph_node_segmentation_mask(
            out,
            threshold=graph_node_segmentation_threshold,
        )
        result = relation_infer(
            h.detach(),
            out,
            model,
            config.MODEL.DECODER.OBJ_TOKEN,
            config.MODEL.DECODER.RLN_TOKEN,
            nms=nms,
            map_=False,
            mode=run.mode,
            distance_weight=distance_weight,
            return_details=run.mode == "vr-mst",
            node_valid_mask=node_valid_mask,
        )
    if len(result) == 3:
        pred_nodes, pred_edges, details = result
        detail = details[0]
    else:
        pred_nodes, pred_edges = result
        detail = None
    return tensor_to_numpy_2d(pred_nodes[0]), edges_to_numpy(pred_edges[0]), detail


def graph_node_segmentation_mask(out: Mapping[str, Any], *, threshold: float) -> torch.Tensor | None:
    """Return a GT-free graph-token mask from predicted segmentation logits.

    TreeFormer graph nodes are normalized ``(x, y)`` coordinates.  Sampling the
    current forward pass' segmentation map at those coordinates ensures graph
    postprocessing cannot retain a node that the same model considers
    background.  A threshold of zero is the explicit compatibility opt-out for
    checkpoints without an auxiliary segmentation head.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"graph node segmentation threshold must be in [0, 1], got {threshold}")
    if threshold == 0.0:
        return None

    aux_maps = out.get("aux_maps")
    pred_nodes = out.get("pred_nodes")
    if not isinstance(aux_maps, torch.Tensor):
        raise ValueError(
            "graph node segmentation gating requires aux_maps; "
            "pass --graph-node-segmentation-threshold 0 to disable it explicitly"
        )
    if not isinstance(pred_nodes, torch.Tensor):
        raise KeyError("model output does not contain pred_nodes")
    if aux_maps.ndim != 4 or aux_maps.shape[1] < 1:
        raise ValueError(f"aux_maps must have shape [B, C>=1, H, W], got {tuple(aux_maps.shape)}")
    if pred_nodes.ndim != 3 or pred_nodes.shape[0] != aux_maps.shape[0] or pred_nodes.shape[-1] < 2:
        raise ValueError(
            "pred_nodes must have shape [B, N, >=2] matching aux_maps batch: "
            f"got {tuple(pred_nodes.shape)} and {tuple(aux_maps.shape)}"
        )

    normalized_nodes = pred_nodes[..., :2].detach().clamp(0.0, 1.0)
    grid = normalized_nodes.mul(2.0).sub(1.0).unsqueeze(2)
    segmentation_confidence = torch.sigmoid(aux_maps[:, :1].detach())
    sampled_confidence = F.grid_sample(
        segmentation_confidence,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).squeeze(-1)
    return sampled_confidence >= threshold


def tensor_to_numpy_2d(nodes: torch.Tensor | np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    if isinstance(nodes, torch.Tensor):
        nodes = nodes.detach().cpu().numpy()
    array = np.asarray(nodes, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return array.reshape(-1, 2)


def edges_to_numpy(edges: torch.Tensor | np.ndarray | Iterable[Iterable[int]]) -> np.ndarray:
    if isinstance(edges, torch.Tensor):
        edges = edges.detach().cpu().numpy()
    array = np.asarray(edges, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return array.reshape(-1, 2)


def add_label(image: Image.Image, label: str, label_height: int = 34) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    canvas = Image.new("RGB", (width, height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    x = max(4, (width - text_width) // 2)
    y = max(2, (label_height - (text_bbox[3] - text_bbox[1])) // 2)
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    return canvas


def draw_graph_overlay(
    image: Image.Image,
    nodes: np.ndarray,
    edges: np.ndarray,
    label: str,
    *,
    edge_width: int = 3,
    node_radius: int = 4,
    inset: bool = False,
    inset_fraction: float = 0.32,
    inset_scale: float = 1.65,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    nodes = tensor_to_numpy_2d(nodes)
    edges = edges_to_numpy(edges)
    pixel_nodes = nodes * np.array([width, height], dtype=np.float32) if len(nodes) else np.empty((0, 2))

    valid_edges: list[tuple[int, int]] = []
    for start, end in edges.tolist():
        if 0 <= start < len(pixel_nodes) and 0 <= end < len(pixel_nodes):
            x0, y0 = pixel_nodes[start]
            x1, y1 = pixel_nodes[end]
            draw.line((float(x0), float(y0), float(x1), float(y1)), fill=(220, 30, 30), width=edge_width)
            valid_edges.append((int(start), int(end)))

    degree = np.zeros((len(pixel_nodes),), dtype=np.int32)
    for start, end in valid_edges:
        degree[start] += 1
        degree[end] += 1

    for index, (x, y) in enumerate(pixel_nodes):
        is_keypoint = degree[index] != 2
        radius = node_radius + (1 if is_keypoint else 0)
        fill = (0, 210, 230) if is_keypoint else (245, 210, 30)
        outline = (25, 90, 110) if is_keypoint else (130, 100, 0)
        draw.ellipse(
            (float(x - radius), float(y - radius), float(x + radius), float(y + radius)),
            fill=fill,
            outline=outline,
            width=1,
        )

    if inset:
        canvas = add_inset(canvas, nodes, edges, fraction=inset_fraction, scale=inset_scale)
    return add_label(canvas, label)


def add_inset(
    image: Image.Image,
    nodes: np.ndarray,
    edges: np.ndarray,
    *,
    fraction: float = 0.32,
    scale: float = 1.65,
) -> Image.Image:
    if len(nodes) == 0:
        return image
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    pixel_nodes = nodes * np.array([width, height], dtype=np.float32)

    degree = np.zeros((len(pixel_nodes),), dtype=np.int32)
    for start, end in edges_to_numpy(edges).tolist():
        if 0 <= start < len(pixel_nodes) and 0 <= end < len(pixel_nodes):
            degree[start] += 1
            degree[end] += 1
    center = pixel_nodes[int(np.argmax(degree))] if degree.max() > 0 else pixel_nodes.mean(axis=0)

    crop_w = max(8, int(width * min(max(fraction, 0.1), 0.8)))
    crop_h = max(8, int(height * min(max(fraction, 0.1), 0.8)))
    left = min(max(int(round(float(center[0]) - crop_w / 2)), 0), max(0, width - crop_w))
    top = min(max(int(round(float(center[1]) - crop_h / 2)), 0), max(0, height - crop_h))
    right = min(width, left + crop_w)
    bottom = min(height, top + crop_h)
    border = (190, 130, 230)
    draw.rectangle((left, top, right, bottom), outline=border, width=2)

    crop = canvas.crop((left, top, right, bottom))
    inset_w = min(width - 8, max(8, int(crop.size[0] * min(max(scale, 1.0), 4.0))))
    inset_h = min(height - 8, max(8, int(crop.size[1] * min(max(scale, 1.0), 4.0))))
    crop = crop.resize((inset_w, inset_h), resample=_resampling_bilinear())
    margin = 8
    paste_x = margin
    paste_y = max(margin, height - inset_h - margin)
    if left < paste_x + inset_w and top < paste_y + inset_h and right > paste_x and bottom > paste_y:
        paste_x = max(margin, width - inset_w - margin)
    canvas.paste(crop, (paste_x, paste_y))
    draw.rectangle((paste_x, paste_y, paste_x + inset_w, paste_y + inset_h), outline=border, width=3)
    return canvas


def make_panel_grid(
    panels: list[Image.Image],
    *,
    columns: int | None,
    pad: int,
    panel_width: int | None,
) -> Image.Image:
    if not panels:
        raise ValueError("no panels to render")
    if panel_width is not None and panel_width > 0:
        normalized = []
        for panel in panels:
            scale = panel_width / float(panel.size[0])
            normalized.append(
                panel.resize((panel_width, max(1, round(panel.size[1] * scale))), resample=_resampling_bilinear())
            )
        panels = normalized

    cell_width = max(panel.size[0] for panel in panels)
    cell_height = max(panel.size[1] for panel in panels)
    columns = len(panels) if columns is None or columns <= 0 else min(columns, len(panels))
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB",
        (columns * cell_width + (columns + 1) * pad, rows * cell_height + (rows + 1) * pad),
        "white",
    )
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        cell = ImageOps.pad(panel, (cell_width, cell_height), method=_resampling_bilinear(), color="white")
        canvas.paste(cell, (pad + col * (cell_width + pad), pad + row * (cell_height + pad)))
    return canvas


def try_load_ground_truth(
    image_path: Path, gt_dir: Path | None, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    candidates: list[Path] = []
    for directory in [gt_dir, image_path.parent]:
        if directory is None:
            continue
        candidates.extend(
            [
                directory / f"{image_path.stem}_annotation.json",
                directory / f"{image_path.stem}.json",
                directory / f"{image_path.stem}.pt",
            ]
        )

    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".json":
            parsed = load_guyot_json(candidate, image_size)
        elif candidate.suffix.lower() == ".pt":
            parsed = load_legacy_pt(candidate)
        else:
            parsed = None
        if parsed is not None:
            return parsed
    return None


def load_guyot_json(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        width, height = image_size
        with path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        features = annotation["VineImage"][0]["VineFeature"][0]
        id_to_index: dict[int, int] = {}
        nodes: list[list[float]] = []
        for index, feature in enumerate(features):
            feature_id = int(feature["FeatureID"])
            x, y = feature["FeatureCoordinates"]
            id_to_index[feature_id] = index
            nodes.append([float(x) / width, float(y) / height])
        edges: list[list[int]] = []
        for index, feature in enumerate(features):
            parent_id = feature.get("ParentID")
            feature_id = feature.get("FeatureID")
            if parent_id is None or parent_id == feature_id:
                continue
            if int(parent_id) in id_to_index:
                edges.append([id_to_index[int(parent_id)], index])
        return np.asarray(nodes, dtype=np.float32), edges_to_numpy(edges)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to load Guyot JSON ground truth {path}: {exc}")
        return None


def load_legacy_pt(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        datapoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(datapoint, Mapping):
            nodes = datapoint["list_DETR_points_left_up"]
            edges = datapoint["DETR_node_collections"]
        else:
            nodes = datapoint.list_DETR_points_left_up
            edges = datapoint.DETR_node_collections
        return tensor_to_numpy_2d(nodes), edges_to_numpy(edges)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to load legacy PT ground truth {path}: {exc}")
        return None


def save_graph_json(path: Path, predictions: Mapping[str, Any]) -> None:
    payload = {}
    for label, prediction in predictions.items():
        if isinstance(prediction, Mapping):
            nodes = prediction["nodes"]
            edges = prediction["edges"]
            details = prediction.get("details")
        else:
            nodes, edges = prediction
            details = None
        payload[label] = {
            "nodes_xy_normalized": tensor_to_numpy_2d(nodes).round(6).tolist(),
            "edges_node_indices": edges_to_numpy(edges).astype(int).tolist(),
        }
        if isinstance(details, Mapping):
            payload[label]["postprocessor_mode"] = details.get("postprocessor_mode")
            for key in ("root_edges_node_indices", "component_id", "augmented_edges"):
                value = details.get(key)
                if value is not None:
                    payload[label][key] = np.asarray(value).astype(int).tolist()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def make_aux_diagnostic_panels(
    *,
    image: torch.Tensor,
    gt_segmentation: torch.Tensor,
    gt_heatmap: torch.Tensor,
    gt_paf: torch.Tensor,
    prediction: Mapping[str, torch.Tensor] | None,
    show_heatmap: bool,
    show_paf: bool,
) -> list[Image.Image]:
    """Build compact dense-aux diagnostics for the graph summary panel.

    GT targets are useful even when the graph checkpoint has no aux head.
    Prediction panels are added only when an aux checkpoint/model is supplied.
    """

    from infer_aux_panel_treeformer import (
        heatmap_to_pil,
        image_tensor_to_pil,
        mask_paf_by_segmentation,
        mask_scalar_map_by_segmentation,
        overlay_map,
        paf_to_rgb,
    )

    input_image = image_tensor_to_pil(image)
    gt_segmentation = gt_segmentation.float().clamp(0.0, 1.0)
    gt_heatmap = gt_heatmap.float().clamp(0.0, 1.0)
    gt_paf = (
        gt_paf.float().permute(2, 0, 1).clamp(-1.0, 1.0)
        if gt_paf.ndim == 3 and gt_paf.shape[-1] == 2
        else gt_paf.float().clamp(-1.0, 1.0)
    )

    panels = [add_label(overlay_map(input_image, gt_segmentation, (40, 210, 80)), "GT segmentation")]
    if prediction is not None:
        panels.append(
            add_label(overlay_map(input_image, prediction["segmentation"], (240, 80, 40)), "Pred segmentation")
        )

    if show_heatmap:
        panels.append(
            add_label(heatmap_to_pil(mask_scalar_map_by_segmentation(gt_heatmap, gt_segmentation)), "GT node heatmap")
        )
        if prediction is not None:
            panels.append(
                add_label(
                    heatmap_to_pil(mask_scalar_map_by_segmentation(prediction["heatmap"], prediction["segmentation"])),
                    "Pred node heatmap",
                )
            )

    if show_paf:
        panels.append(add_label(paf_to_rgb(mask_paf_by_segmentation(gt_paf, gt_segmentation)), "GT edge direction"))
        if prediction is not None:
            panels.append(
                add_label(
                    paf_to_rgb(mask_paf_by_segmentation(prediction["paf"], prediction["segmentation"])),
                    "Pred edge direction",
                )
            )

    return panels


def build_aux_diagnostic_context(
    *,
    split_root: Path,
    checkpoint_path: Path | None,
    max_size: int,
    loader: str,
    weights: str,
    legacy_key: str,
    strict: bool,
    device: torch.device,
    show_untrained_maps: bool,
) -> AuxDiagnosticContext:
    from infer_aux_panel_treeformer import (
        build_aux_panel_dataset,
        checkpoint_train_weight,
        load_aux_model,
        load_checkpoint_mapping,
        unpack_aux_panel_sample,
    )

    checkpoint: Mapping[str, Any] = {"config": {}}
    model = None
    show_heatmap = True
    show_paf = True
    if checkpoint_path is not None:
        checkpoint = load_checkpoint_mapping(checkpoint_path)
        show_heatmap = show_untrained_maps or checkpoint_train_weight(checkpoint, "W_AUX_HEATMAP", 1.0) > 0.0
        show_paf = show_untrained_maps or checkpoint_train_weight(checkpoint, "W_AUX_PAF", 1.0) > 0.0
        model = load_aux_model(
            checkpoint_path=checkpoint_path,
            device=device,
            weights=weights,
            legacy_key=legacy_key,
            strict=strict,
        )

    dataset, loader_name = build_aux_panel_dataset(
        split_root=split_root,
        checkpoint=checkpoint,
        max_size=max_size,
        loader=loader,
    )
    samples_by_name: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for index in range(len(dataset)):
        image, _label, pafs, segmentation, heatmap, _sample_id, sample_name = unpack_aux_panel_sample(dataset[index])
        samples_by_name[str(sample_name)] = (image, pafs, segmentation, heatmap)

    return AuxDiagnosticContext(
        samples_by_name=samples_by_name,
        model=model,
        device=device,
        show_heatmap=show_heatmap,
        show_paf=show_paf,
        loader_name=loader_name,
    )


def predict_aux_diagnostics(
    context: AuxDiagnosticContext,
    image: torch.Tensor,
    target_size: tuple[int, int],
) -> dict[str, torch.Tensor] | None:
    if context.model is None:
        return None
    from infer_aux_panel_treeformer import predict_aux_maps

    return predict_aux_maps(context.model, image, context.device, target_size)


def _resampling_bilinear() -> int:
    return getattr(Image, "Resampling", Image).BILINEAR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TreeFormer per-image inference panel renderer")
    parser.add_argument(
        "--config", default=None, help="Default legacy YAML config for runs without embedded Hydra config"
    )
    parser.add_argument("--legacy-split-root", default=None, help="Optional split root with img/ and data/ directories")
    parser.add_argument("--image-dir", default=None, help="Directory containing input images")
    parser.add_argument(
        "--gt-dir", default=None, help="Directory containing *_annotation.json, .json, or .pt ground truth"
    )
    parser.add_argument("--output-dir", required=True, help="Directory where panels are saved")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        help="Run spec: 'LABEL|CHECKPOINT|MODE' or 'LABEL|CONFIG|CHECKPOINT|MODE'. MODE is raw, mst, mst-dist, or vr-mst.",
    )
    parser.add_argument(
        "--weights", default="auto", choices=("auto", "ema", "model"), help="Checkpoint weights to load"
    )
    parser.add_argument("--legacy-key", default="net", help="Legacy checkpoint state_dict key")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="Inference device")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--max-size", type=int, default=None, help="Resize longest side for model input")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to render")
    parser.add_argument("--nms", action="store_true", help="Apply token NMS before relation inference")
    parser.add_argument(
        "--graph-node-segmentation-threshold",
        type=float,
        default=0.5,
        help=(
            "Keep graph-node candidates only where this model's predicted segmentation confidence meets the threshold. "
            "Use 0 only as an explicit compatibility opt-out for checkpoints without an aux segmentation head."
        ),
    )
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading")
    parser.add_argument("--no-gt", action="store_true", help="Do not try to add a Ground truth panel")
    parser.add_argument("--columns", type=int, default=0, help="Grid columns; 0 means one horizontal row")
    parser.add_argument("--panel-width", type=int, default=420, help="Rendered width of each panel cell; 0 keeps size")
    parser.add_argument("--pad", type=int, default=8, help="Whitespace between panel cells")
    parser.add_argument("--edge-width", type=int, default=3, help="Overlay edge width")
    parser.add_argument("--node-radius", type=int, default=4, help="Overlay node radius")
    parser.add_argument("--distance-weight", type=float, default=0.5, help="Distance weight for mode=mst-dist")
    parser.add_argument("--inset", action="store_true", help="Add zoom inset around densest/highest-degree region")
    parser.add_argument("--inset-fraction", type=float, default=0.32, help="Inset crop fraction")
    parser.add_argument("--inset-scale", type=float, default=1.65, help="Inset zoom scale")
    parser.add_argument("--save-graph-json", action="store_true", help="Save predicted nodes/edges JSON")
    parser.add_argument(
        "--include-aux-maps",
        action="store_true",
        help="Append dense aux target maps from --legacy-split-root to each graph summary panel.",
    )
    parser.add_argument(
        "--aux-checkpoint",
        default=None,
        help="Optional aux-supervised checkpoint. When provided, predicted segmentation/heatmap/edge-direction panels are appended.",
    )
    parser.add_argument(
        "--aux-weights",
        default="auto",
        choices=("auto", "ema", "model"),
        help="Checkpoint weights to load for --aux-checkpoint.",
    )
    parser.add_argument("--aux-legacy-key", default="net", help="Legacy checkpoint state_dict key for --aux-checkpoint")
    parser.add_argument("--aux-strict", action="store_true", help="Use strict aux checkpoint loading")
    parser.add_argument(
        "--aux-loader",
        default="auto",
        choices=("auto", "legacy", "fast-seg"),
        help="Target loader for dense aux maps. auto uses the aux checkpoint DATA.FAST_SEGMENTATION_LOADER flag.",
    )
    parser.add_argument(
        "--aux-max-size",
        type=int,
        default=None,
        help="Max image size for dense aux target/prediction panels. Defaults to --max-size, then 128.",
    )
    parser.add_argument(
        "--aux-show-untrained-maps",
        action="store_true",
        help="Render aux heatmap/edge-direction predictions even when the aux checkpoint loss weights are zero.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    image_dir = Path(args.image_dir) if args.image_dir else None
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    if args.legacy_split_root:
        split_root = Path(args.legacy_split_root)
        image_dir = image_dir or split_root / "img"
        gt_dir = gt_dir or split_root / "data"
    if image_dir is None:
        raise ValueError("--image-dir or --legacy-split-root is required")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    default_config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    models: dict[str, torch.nn.Module] = {}
    configs: dict[str, AttrDict] = {}
    for run in args.run:
        checkpoint = load_checkpoint_mapping(run.checkpoint)
        config = load_config_for_run(
            config_path=run.config, default_config_path=default_config_path, checkpoint=checkpoint
        )
        configs[run.label] = config
        models[run.label] = load_model(
            config,
            run.checkpoint,
            device=device,
            strict=bool(args.strict),
            weights=str(args.weights),
            legacy_key=str(args.legacy_key),
        )
        print(f"  {run.label}: {run.checkpoint} mode={run.mode} weights={args.weights}")

    aux_context: AuxDiagnosticContext | None = None
    if args.include_aux_maps or args.aux_checkpoint:
        if not args.legacy_split_root:
            raise ValueError("--include-aux-maps and --aux-checkpoint require --legacy-split-root")
        aux_checkpoint = Path(args.aux_checkpoint) if args.aux_checkpoint else None
        if aux_checkpoint is not None and not aux_checkpoint.is_file():
            raise FileNotFoundError(f"aux checkpoint not found: {aux_checkpoint}")
        aux_context = build_aux_diagnostic_context(
            split_root=Path(args.legacy_split_root),
            checkpoint_path=aux_checkpoint,
            max_size=int(args.aux_max_size or args.max_size or 128),
            loader=str(args.aux_loader),
            weights=str(args.aux_weights),
            legacy_key=str(args.aux_legacy_key),
            strict=bool(args.aux_strict),
            device=device,
            show_untrained_maps=bool(args.aux_show_untrained_maps),
        )
        prediction_state = "with predictions" if aux_context.model is not None else "targets only"
        print(
            f"  Aux diagnostics: {len(aux_context.samples_by_name)} samples via {aux_context.loader_name} loader ({prediction_state})"
        )

    image_paths = discover_images(image_dir, recursive=bool(args.recursive))
    if args.limit is not None and args.limit > 0:
        image_paths = image_paths[: args.limit]
    print(f"Found {len(image_paths)} images.")
    for index, image_path in enumerate(image_paths, start=1):
        original_image, tensor = load_image_for_model(image_path, max_size=args.max_size)
        panels = [add_label(original_image, "Input image")]
        predictions: dict[str, Any] = {}

        if not args.no_gt:
            gt = try_load_ground_truth(image_path, gt_dir, image_size=original_image.size)
            if gt is not None:
                panels.append(
                    draw_graph_overlay(
                        original_image,
                        gt[0],
                        gt[1],
                        "Ground truth",
                        edge_width=args.edge_width,
                        node_radius=args.node_radius,
                        inset=args.inset,
                        inset_fraction=args.inset_fraction,
                        inset_scale=args.inset_scale,
                    )
                )

        for run in args.run:
            nodes, edges, details = run_inference(
                model=models[run.label],
                config=configs[run.label],
                tensor=tensor,
                device=device,
                run=run,
                nms=bool(args.nms),
                distance_weight=float(args.distance_weight),
                graph_node_segmentation_threshold=float(args.graph_node_segmentation_threshold),
            )
            predictions[run.label] = {"nodes": nodes, "edges": edges, "details": details}
            panels.append(
                draw_graph_overlay(
                    original_image,
                    nodes,
                    edges,
                    run.label,
                    edge_width=args.edge_width,
                    node_radius=args.node_radius,
                    inset=args.inset,
                    inset_fraction=args.inset_fraction,
                    inset_scale=args.inset_scale,
                )
            )

        if aux_context is not None:
            aux_sample = aux_context.samples_by_name.get(image_path.stem)
            if aux_sample is None:
                print(f"[WARN] no aux-map sample found for image stem: {image_path.stem}")
            else:
                aux_image, aux_pafs, aux_segmentation, aux_heatmap = aux_sample
                aux_prediction = predict_aux_diagnostics(
                    aux_context,
                    aux_image,
                    target_size=(int(aux_segmentation.shape[-2]), int(aux_segmentation.shape[-1])),
                )
                panels.extend(
                    make_aux_diagnostic_panels(
                        image=aux_image,
                        gt_segmentation=aux_segmentation,
                        gt_heatmap=aux_heatmap,
                        gt_paf=aux_pafs,
                        prediction=aux_prediction,
                        show_heatmap=aux_context.show_heatmap,
                        show_paf=aux_context.show_paf,
                    )
                )

        panel = make_panel_grid(
            panels,
            columns=None if int(args.columns) == 0 else int(args.columns),
            pad=int(args.pad),
            panel_width=None if int(args.panel_width) == 0 else int(args.panel_width),
        )
        save_path = output_dir / f"{image_path.stem}_panel.png"
        panel.save(save_path)
        if args.save_graph_json:
            save_graph_json(output_dir / f"{image_path.stem}_pred_graph.json", predictions)
        print(f"[{index}/{len(image_paths)}] saved: {save_path}")


if __name__ == "__main__":
    main()
