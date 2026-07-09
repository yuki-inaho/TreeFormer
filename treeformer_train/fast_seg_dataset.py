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
from .detail_targets import make_stdc_detail_boundary_target


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class FastSegSample:
    sample_id: str
    data_path: Path
    image_path: Path
    mask_path: Path


def _load_graph_annotation(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    datapoint = torch.load(path, map_location="cpu", weights_only=False)
    nodes = torch.as_tensor(datapoint.list_DETR_points_left_up, dtype=torch.float32)
    edges = torch.as_tensor(datapoint.DETR_node_collections, dtype=torch.long)
    return nodes, edges


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
            raise FileNotFoundError(f"missing segmentation mask for sample {sample_id!r} under {[str(item) for item in mask_dirs]}")
        samples.append(FastSegSample(sample_id=sample_id, data_path=data_path, image_path=image_path, mask_path=mask_path))
    return samples


def _cache_key(sample: FastSegSample, *, max_size: int, detail_config: dict[str, Any]) -> str:
    payload = {
        "sample_id": sample.sample_id,
        "data_name": sample.data_path.name,
        "image_name": sample.image_path.name,
        "mask_name": sample.mask_path.name,
        "max_size": int(max_size),
        "detail": detail_config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def build_fast_seg_cache(
    *,
    split_root: str | Path,
    cache_dir: str | Path,
    max_size: int,
    detail_threshold: float = 0.1,
    detail_scales: tuple[int, ...] = (1, 2, 4),
    detail_support_kernel_size: int = 3,
    resize_policy: str = "legacy_half",
    overwrite: bool = False,
) -> dict[str, int]:
    samples = discover_fast_seg_samples(split_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    detail_config = {
        "threshold": float(detail_threshold),
        "scales": [int(item) for item in detail_scales],
        "support_kernel_size": int(detail_support_kernel_size),
    }
    if str(resize_policy).lower() != "legacy_half":
        detail_config["resize_policy"] = str(resize_policy).lower()
    written = 0
    skipped = 0
    for sample in samples:
        cache_path = cache_dir / f"{_cache_key(sample, max_size=max_size, detail_config=detail_config)}.pt"
        if cache_path.exists() and not overwrite:
            skipped += 1
            continue
        image, segmentation = _prepare_image_and_mask(
            sample.image_path,
            sample.mask_path,
            max_size,
            resize_policy=resize_policy,
        )
        detail = make_stdc_detail_boundary_target(
            segmentation,
            threshold=detail_threshold,
            scales=detail_scales,
            support_kernel_size=detail_support_kernel_size,
        ).squeeze(0).squeeze(0)
        nodes, edges = _load_graph_annotation(sample.data_path)
        payload = {
            "sample_id": sample.sample_id,
            "image": image,
            "segmentation": segmentation,
            "detail_boundary": detail.contiguous(),
            "nodes": nodes,
            "edges": edges,
            "detail_config": detail_config,
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
        detail_threshold: float = 0.1,
        detail_scales: tuple[int, ...] = (1, 2, 4),
        detail_support_kernel_size: int = 3,
        resize_policy: str = "legacy_half",
        aux_target_mode: str = "seg_only",
        heatmap_sigma: float = 3.0,
        heatmap_cutoff: float = 0.01,
        paf_line_thickness: int = 2,
        paf_mask_thickness: int = 6,
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
        self.detail_threshold = float(detail_threshold)
        self.detail_scales = tuple(int(item) for item in detail_scales)
        self.detail_support_kernel_size = int(detail_support_kernel_size)
        self.resize_policy = str(resize_policy).lower()
        if self.resize_policy not in {"legacy_half", "full"}:
            raise ValueError(f"resize_policy must be one of ['legacy_half', 'full'], got {resize_policy!r}")
        self.aux_target_config: AuxMapTargetConfig = make_aux_map_target_config(
            mode=aux_target_mode,
            heatmap_sigma=heatmap_sigma,
            heatmap_cutoff=heatmap_cutoff,
            paf_line_thickness=paf_line_thickness,
            paf_mask_thickness=paf_mask_thickness,
        )
        self.detail_config = {
            "threshold": self.detail_threshold,
            "scales": list(self.detail_scales),
            "support_kernel_size": self.detail_support_kernel_size,
        }
        if self.resize_policy != "legacy_half":
            self.detail_config["resize_policy"] = self.resize_policy
        self._memory_cache: dict[int, tuple[Any, ...]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, sample: FastSegSample) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured")
        return self.cache_dir / f"{_cache_key(sample, max_size=self.max_size, detail_config=self.detail_config)}.pt"

    def _load_from_disk_cache(self, sample: FastSegSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_path = self._cache_path(sample)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"missing fast segmentation cache file: {cache_path}. "
                "Run generate_fast_seg_cache.py before using DATA.SEG_CACHE_MODE=disk."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return (
            payload["image"].float(),
            payload["segmentation"].float(),
            payload["detail_boundary"].float(),
            payload["nodes"].float(),
            payload["edges"].long(),
        )

    def _build_sample(self, sample: FastSegSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image, segmentation = _prepare_image_and_mask(
            sample.image_path,
            sample.mask_path,
            self.max_size,
            resize_policy=self.resize_policy,
        )
        detail = make_stdc_detail_boundary_target(
            segmentation,
            threshold=self.detail_threshold,
            scales=self.detail_scales,
            support_kernel_size=self.detail_support_kernel_size,
        ).squeeze(0).squeeze(0)
        nodes, edges = _load_graph_annotation(sample.data_path)
        return image, segmentation, detail.contiguous(), nodes, edges

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        if index in self._memory_cache:
            return self._memory_cache[index]
        sample = self.samples[index]
        if self.cache_mode == "disk":
            image, segmentation, detail, nodes, edges = self._load_from_disk_cache(sample)
        else:
            image, segmentation, detail, nodes, edges = self._build_sample(sample)

        height, width = int(segmentation.shape[-2]), int(segmentation.shape[-1])
        pafs, paf_mask, heatmap = make_aux_map_targets(
            nodes,
            edges,
            image_size=(height, width),
            config=self.aux_target_config,
        )
        item = (
            image.contiguous(),
            sample.sample_id,
            nodes,
            edges,
            pafs,
            paf_mask,
            segmentation.contiguous(),
            heatmap,
            sample.data_path.name,
        )
        if self.cache_mode == "memory":
            self._memory_cache[index] = item
        return item


def _parse_scales(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repo-external cache files for FastSegSupervisedDataset")
    parser.add_argument("--dataset-root", required=True, help="Legacy TreeFormer dataset root containing split directories")
    parser.add_argument("--cache-root", required=True, help="Directory where cache files will be written")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Split names to cache")
    parser.add_argument("--max-size", type=int, default=128, help="Model input max size")
    parser.add_argument("--detail-threshold", type=float, default=0.1)
    parser.add_argument("--detail-scales", default="1,2,4")
    parser.add_argument("--detail-support-kernel-size", type=int, default=3)
    parser.add_argument("--resize-policy", choices=["legacy_half", "full"], default="legacy_half")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_root = Path(args.cache_root)
    scales = _parse_scales(str(args.detail_scales))
    totals = {"samples": 0, "written": 0, "skipped": 0}
    for split in args.splits:
        stats = build_fast_seg_cache(
            split_root=dataset_root / split,
            cache_dir=cache_root / split,
            max_size=int(args.max_size),
            detail_threshold=float(args.detail_threshold),
            detail_scales=scales,
            detail_support_kernel_size=int(args.detail_support_kernel_size),
            resize_policy=str(args.resize_policy),
            overwrite=bool(args.overwrite),
        )
        for key, value in stats.items():
            totals[key] += value
        print(f"{split}: {stats}")
    print(f"total: {totals}")


if __name__ == "__main__":
    main()
