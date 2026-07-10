from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .aux_map_targets import AuxMapTargetConfig, make_aux_map_target_config, make_aux_map_targets
from .virtual_root import load_forest_metadata


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
CACHE_FORMAT_VERSION = 3


@dataclass(frozen=True)
class FastSegSample:
    sample_id: str
    data_path: Path
    image_path: Path
    mask_path: Path


def _load_graph_annotation(
    path: Path, *, strict_virtual_root_metadata: bool = False
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    datapoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(datapoint, dict):
        raw_nodes = datapoint["list_DETR_points_left_up"]
        raw_edges = datapoint["DETR_node_collections"]
    else:
        raw_nodes = datapoint.list_DETR_points_left_up
        raw_edges = datapoint.DETR_node_collections
    nodes = torch.as_tensor(raw_nodes, dtype=torch.float32)
    edges = torch.as_tensor(raw_edges, dtype=torch.long)
    metadata = load_forest_metadata(
        datapoint,
        nodes=nodes,
        edges=edges,
        strict_virtual_root=bool(strict_virtual_root_metadata),
    )
    return nodes, edges, metadata


def _find_existing_path(directory: Path, sample_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"failed to read segmentation mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_float = mask.astype(np.float32)
    if mask_float.max(initial=0.0) > 1.0:
        return (mask_float > 127.0).astype(np.float32)
    return (mask_float > 0.5).astype(np.float32)


def _resize_hw(height: int, width: int, max_size: int, resize_policy: str = "legacy_half") -> tuple[int, int]:
    resize_policy = resize_policy.lower()
    if resize_policy == "legacy_half":
        height = height // 2
        width = width // 2
    elif resize_policy != "full":
        raise ValueError(f"resize_policy must be one of ['legacy_half', 'full'], got {resize_policy!r}")
    cut_height = height
    cut_width = width
    if max(cut_width, cut_height) <= int(max_size):
        return cut_height, cut_width
    if cut_width > cut_height:
        scale = float(max_size) / float(cut_width)
        return int(cut_height * scale), int(max_size)
    scale = float(max_size) / float(cut_height)
    return int(max_size), int(cut_width * scale)


def _prepare_image_and_mask(
    image_path: Path,
    mask_path: Path,
    max_size: int,
    resize_policy: str = "legacy_half",
) -> tuple[torch.Tensor, torch.Tensor]:
    image = _read_rgb(image_path)
    mask = _read_binary_mask(mask_path)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    out_h, out_w = _resize_hw(image.shape[0], image.shape[1], max_size, resize_policy=resize_policy)
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    image_tensor = image_tensor.sub(0.5).div(0.5)
    mask_tensor = torch.from_numpy((mask > 0.5).astype(np.float32)).contiguous()
    return image_tensor, mask_tensor


def discover_fast_seg_samples(split_root: str | Path) -> list[FastSegSample]:
    split_root = Path(split_root)
    data_dir = split_root / "data"
    image_dir = split_root / "img"
    mask_dirs = (split_root / "seg", split_root / "unet")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data directory: {data_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {image_dir}")

    samples = []
    for data_path in sorted(data_dir.glob("*.pt")):
        sample_id = data_path.stem
        image_path = _find_existing_path(image_dir, sample_id)
        if image_path is None:
            raise FileNotFoundError(f"missing RGB image for sample {sample_id!r} under {image_dir}")
        mask_path = None
        for mask_dir in mask_dirs:
            if mask_dir.is_dir():
                mask_path = _find_existing_path(mask_dir, sample_id)
                if mask_path is not None:
                    break
        if mask_path is None:
            raise FileNotFoundError(
                f"missing segmentation mask for sample {sample_id!r} under {[str(item) for item in mask_dirs]}"
            )
        samples.append(
            FastSegSample(sample_id=sample_id, data_path=data_path, image_path=image_path, mask_path=mask_path)
        )
    return samples


def _cache_config(*, resize_policy: str, aux_target_config: AuxMapTargetConfig) -> dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "resize_policy": str(resize_policy).lower(),
        "aux_target": {
            "mode": aux_target_config.mode,
            "heatmap_sigma": aux_target_config.heatmap_sigma,
            "heatmap_cutoff": aux_target_config.heatmap_cutoff,
            "paf_line_thickness": aux_target_config.paf_line_thickness,
            "paf_mask_thickness": aux_target_config.paf_mask_thickness,
            "direction_target_source": aux_target_config.direction_target_source,
            "direction_encoding": aux_target_config.direction_encoding,
            "direction_tangent_radius": aux_target_config.direction_tangent_radius,
            "direction_junction_exclusion_radius": aux_target_config.direction_junction_exclusion_radius,
        },
    }


def _cache_key(sample: FastSegSample, *, max_size: int, cache_config: dict[str, Any]) -> str:
    payload = {
        "sample_id": sample.sample_id,
        "data_name": sample.data_path.name,
        "image_name": sample.image_path.name,
        "mask_name": sample.mask_path.name,
        "max_size": int(max_size),
        "cache": cache_config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def build_fast_seg_cache(
    *,
    split_root: str | Path,
    cache_dir: str | Path,
    max_size: int,
    resize_policy: str = "legacy_half",
    aux_target_mode: str = "seg_only",
    heatmap_sigma: float = 3.0,
    heatmap_cutoff: float = 0.01,
    paf_line_thickness: int = 2,
    paf_mask_thickness: int = 6,
    direction_target_source: str = "graph_edges",
    direction_encoding: str = "vector",
    direction_tangent_radius: int = 8,
    direction_junction_exclusion_radius: int = 6,
    strict_virtual_root_metadata: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    samples = discover_fast_seg_samples(split_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    aux_target_config = make_aux_map_target_config(
        mode=aux_target_mode,
        heatmap_sigma=heatmap_sigma,
        heatmap_cutoff=heatmap_cutoff,
        paf_line_thickness=paf_line_thickness,
        paf_mask_thickness=paf_mask_thickness,
        direction_target_source=direction_target_source,
        direction_encoding=direction_encoding,
        direction_tangent_radius=direction_tangent_radius,
        direction_junction_exclusion_radius=direction_junction_exclusion_radius,
    )
    cache_config = _cache_config(resize_policy=resize_policy, aux_target_config=aux_target_config)
    written = 0
    skipped = 0
    for sample in samples:
        cache_path = cache_dir / f"{_cache_key(sample, max_size=max_size, cache_config=cache_config)}.pt"
        if cache_path.exists() and not overwrite:
            skipped += 1
            continue
        image, segmentation = _prepare_image_and_mask(
            sample.image_path,
            sample.mask_path,
            max_size,
            resize_policy=resize_policy,
        )
        nodes, edges, metadata = _load_graph_annotation(
            sample.data_path,
            strict_virtual_root_metadata=bool(strict_virtual_root_metadata),
        )
        pafs, paf_mask, heatmap = make_aux_map_targets(
            nodes,
            edges,
            image_size=(int(segmentation.shape[-2]), int(segmentation.shape[-1])),
            config=aux_target_config,
            segmentation=segmentation,
        )
        payload = {
            "sample_id": sample.sample_id,
            "cache_format_version": CACHE_FORMAT_VERSION,
            "cache_config": cache_config,
            "image": image,
            "segmentation": segmentation,
            "nodes": nodes,
            "edges": edges,
            "paf": pafs,
            "paf_mask": paf_mask,
            "heatmap": heatmap,
            "forest_metadata": metadata,
            "max_size": int(max_size),
        }
        tmp_path = cache_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        written += 1
    return {"samples": len(samples), "written": written, "skipped": skipped}


class FastSegSupervisedDataset(Dataset):
    """TreeFormer-format segmentation-only dataset without graph-derived target generation."""

    def __init__(
        self,
        split_root: str | Path,
        *,
        max_size: int,
        cache_mode: str = "none",
        cache_dir: str | Path | None = None,
        resize_policy: str = "legacy_half",
        aux_target_mode: str = "seg_only",
        heatmap_sigma: float = 3.0,
        heatmap_cutoff: float = 0.01,
        paf_line_thickness: int = 2,
        paf_mask_thickness: int = 6,
        direction_target_source: str = "graph_edges",
        direction_encoding: str = "vector",
        direction_tangent_radius: int = 8,
        direction_junction_exclusion_radius: int = 6,
        return_forest_metadata: bool = False,
        strict_virtual_root_metadata: bool = False,
    ) -> None:
        self.split_root = Path(split_root)
        self.max_size = int(max_size)
        self.samples = discover_fast_seg_samples(self.split_root)
        self.cache_mode = str(cache_mode or "none").lower()
        if self.cache_mode not in {"none", "memory", "disk"}:
            raise ValueError(f"cache_mode must be one of ['none', 'memory', 'disk'], got {cache_mode!r}")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_mode == "disk" and self.cache_dir is None:
            raise ValueError("cache_dir is required when cache_mode='disk'")
        self.return_forest_metadata = bool(return_forest_metadata)
        self.strict_virtual_root_metadata = bool(strict_virtual_root_metadata)
        self.resize_policy = str(resize_policy).lower()
        if self.resize_policy not in {"legacy_half", "full"}:
            raise ValueError(f"resize_policy must be one of ['legacy_half', 'full'], got {resize_policy!r}")
        self.aux_target_config: AuxMapTargetConfig = make_aux_map_target_config(
            mode=aux_target_mode,
            heatmap_sigma=heatmap_sigma,
            heatmap_cutoff=heatmap_cutoff,
            paf_line_thickness=paf_line_thickness,
            paf_mask_thickness=paf_mask_thickness,
            direction_target_source=direction_target_source,
            direction_encoding=direction_encoding,
            direction_tangent_radius=direction_tangent_radius,
            direction_junction_exclusion_radius=direction_junction_exclusion_radius,
        )
        self.cache_config = _cache_config(resize_policy=self.resize_policy, aux_target_config=self.aux_target_config)
        self._memory_cache: dict[int, tuple[Any, ...]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, sample: FastSegSample) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured")
        return self.cache_dir / f"{_cache_key(sample, max_size=self.max_size, cache_config=self.cache_config)}.pt"

    def _load_from_disk_cache(
        self,
        sample: FastSegSample,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, Any],
    ]:
        cache_path = self._cache_path(sample)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing fast segmentation cache file: {cache_path}. "
                "Run generate_fast_seg_cache.py before using DATA.SEG_CACHE_MODE=disk."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported fast segmentation cache format in {cache_path}; "
                "regenerate the cache with generate_fast_seg_cache.py."
            )
        return (
            payload["image"].float(),
            payload["segmentation"].float(),
            payload["nodes"].float(),
            payload["edges"].long(),
            payload.get("paf", None).float() if payload.get("paf", None) is not None else None,
            payload.get("paf_mask", None).bool() if payload.get("paf_mask", None) is not None else None,
            payload.get("heatmap", None).float() if payload.get("heatmap", None) is not None else None,
            payload.get("forest_metadata")
            or load_forest_metadata(payload, nodes=payload["nodes"], edges=payload["edges"], strict_virtual_root=False),
        )

    def _build_sample(
        self, sample: FastSegSample
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        image, segmentation = _prepare_image_and_mask(
            sample.image_path,
            sample.mask_path,
            self.max_size,
            resize_policy=self.resize_policy,
        )
        # No detail boundary target is computed here: __getitem__ never places it
        # in the returned tuple, so computing it would be dead weight (see
        # _load_from_disk_cache for the same note).
        nodes, edges, metadata = _load_graph_annotation(
            sample.data_path,
            strict_virtual_root_metadata=self.strict_virtual_root_metadata,
        )
        return image, segmentation, nodes, edges, metadata

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        if index in self._memory_cache:
            return self._memory_cache[index]
        sample = self.samples[index]
        if self.cache_mode == "disk":
            image, segmentation, nodes, edges, cached_pafs, cached_paf_mask, cached_heatmap, metadata = (
                self._load_from_disk_cache(sample)
            )
        else:
            image, segmentation, nodes, edges, metadata = self._build_sample(sample)
            cached_pafs = None
            cached_paf_mask = None
            cached_heatmap = None

        height, width = int(segmentation.shape[-2]), int(segmentation.shape[-1])
        if cached_pafs is None or cached_paf_mask is None or cached_heatmap is None:
            pafs, paf_mask, heatmap = make_aux_map_targets(
                nodes,
                edges,
                image_size=(height, width),
                config=self.aux_target_config,
                segmentation=segmentation,
            )
        else:
            pafs = cached_pafs
            paf_mask = cached_paf_mask
            heatmap = cached_heatmap
        item = (
            image.contiguous(),
            sample.sample_id,
            nodes,
            edges,
            pafs,
            paf_mask,
            segmentation.contiguous(),
            heatmap,
        )
        if self.return_forest_metadata:
            item = item + (
                metadata,
                sample.data_path.name,
            )
        else:
            item = item + (sample.data_path.name,)
        if self.cache_mode == "memory":
            self._memory_cache[index] = item
        return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repo-external cache files for FastSegSupervisedDataset")
    parser.add_argument(
        "--dataset-root", required=True, help="Legacy TreeFormer dataset root containing split directories"
    )
    parser.add_argument("--cache-root", required=True, help="Directory where cache files will be written")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Split names to cache")
    parser.add_argument("--max-size", type=int, default=128, help="Model input max size")
    parser.add_argument("--resize-policy", choices=["legacy_half", "full"], default="legacy_half")
    parser.add_argument("--aux-target-mode", default="seg_only")
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--heatmap-cutoff", type=float, default=0.01)
    parser.add_argument("--paf-line-thickness", type=int, default=2)
    parser.add_argument("--paf-mask-thickness", type=int, default=6)
    parser.add_argument("--direction-target-source", choices=["graph_edges", "mask_skeleton"], default="graph_edges")
    parser.add_argument("--direction-encoding", choices=["vector", "double_angle"], default="vector")
    parser.add_argument("--direction-tangent-radius", type=int, default=8)
    parser.add_argument("--direction-junction-exclusion-radius", type=int, default=6)
    parser.add_argument("--strict-virtual-root-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_root = Path(args.cache_root)
    totals = {"samples": 0, "written": 0, "skipped": 0}
    for split in args.splits:
        stats = build_fast_seg_cache(
            split_root=dataset_root / split,
            cache_dir=cache_root / split,
            max_size=int(args.max_size),
            resize_policy=str(args.resize_policy),
            aux_target_mode=str(args.aux_target_mode),
            heatmap_sigma=float(args.heatmap_sigma),
            heatmap_cutoff=float(args.heatmap_cutoff),
            paf_line_thickness=int(args.paf_line_thickness),
            paf_mask_thickness=int(args.paf_mask_thickness),
            direction_target_source=str(args.direction_target_source),
            direction_encoding=str(args.direction_encoding),
            direction_tangent_radius=int(args.direction_tangent_radius),
            direction_junction_exclusion_radius=int(args.direction_junction_exclusion_radius),
            strict_virtual_root_metadata=bool(args.strict_virtual_root_metadata),
            overwrite=bool(args.overwrite),
        )
        for key, value in stats.items():
            totals[key] += value
        print(f"{split}: {stats}")
    print(f"total: {totals}")


if __name__ == "__main__":
    main()
