from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from train_mst import build_train_val_datasets, custom_collate_fn
from treeformer_train.fast_seg_dataset import FastSegSupervisedDataset, build_fast_seg_cache


def _write_split(root: Path, split: str, *, count: int = 2) -> Path:
    split_root = root / split
    (split_root / "data").mkdir(parents=True)
    (split_root / "img").mkdir()
    (split_root / "seg").mkdir()
    for index in range(count):
        sample_id = f"sample_{index:02d}"
        torch.save(
            SimpleNamespace(
                list_DETR_points_left_up=torch.tensor([[0.15, 0.15], [0.85, 0.85]], dtype=torch.float32),
                DETR_node_collections=torch.tensor([[0, 1]], dtype=torch.long),
            ),
            split_root / "data" / f"{sample_id}.pt",
        )
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        image[:, :, 0] = 64 + index
        image[:, :, 1] = 128
        Image.fromarray(image).save(split_root / "img" / f"{sample_id}.png")
        mask = np.zeros((64, 80), dtype=np.uint8)
        mask[8:56, 12:68] = 255
        Image.fromarray(mask).save(split_root / "seg" / f"{sample_id}.png")
    return split_root


def test_fast_seg_dataset_returns_legacy_aux_batch_contract(tmp_path: Path):
    split_root = _write_split(tmp_path, "train", count=2)
    dataset = FastSegSupervisedDataset(split_root, max_size=64)

    image, sample_id, nodes, edges, pafs, paf_mask, segmentation, heatmap, data_id = dataset[0]

    assert image.shape == (3, 32, 40)
    assert image.dtype == torch.float32
    assert sample_id == "sample_00"
    assert data_id == "sample_00.pt"
    assert nodes.shape == (2, 2)
    assert edges.shape == (1, 2)
    assert pafs.shape == (32, 40, 2)
    assert paf_mask.shape == (32, 40)
    assert segmentation.shape == (32, 40)
    assert heatmap.shape == (32, 40)
    assert torch.equal(torch.unique(segmentation), torch.tensor([0.0, 1.0]))

    batch = custom_collate_fn([dataset[0], dataset[1]])[0]
    assert len(batch[0]) == 2
    assert batch[3].shape == (2, 2, 32, 40)
    assert batch[5].shape == (2, 1, 32, 40)
    assert batch[6].shape == (2, 1, 32, 40)


def test_fast_seg_dataset_uses_disk_cache(tmp_path: Path):
    split_root = _write_split(tmp_path, "train", count=1)
    cache_dir = tmp_path / "cache" / "train"

    stats = build_fast_seg_cache(split_root=split_root, cache_dir=cache_dir, max_size=64)
    dataset = FastSegSupervisedDataset(split_root, max_size=64, cache_mode="disk", cache_dir=cache_dir)

    assert stats == {"samples": 1, "written": 1, "skipped": 0}
    assert len(list(cache_dir.glob("*.pt"))) == 1
    image, *_rest = dataset[0]
    assert image.shape == (3, 32, 40)


def test_build_train_val_datasets_can_select_fast_seg_loader(tmp_path: Path):
    _write_split(tmp_path, "train", count=2)
    _write_split(tmp_path, "val", count=1)
    data_config = SimpleNamespace(
        DATASET="treeformer-2D",
        DATA_PATH=str(tmp_path),
        MAX_SIZE=64,
        TRAIN_LIMIT=None,
        VAL_LIMIT=None,
        AUGMENTATION=SimpleNamespace(enabled=False),
        LEGACY_ROTATE=False,
        SEGMENTATION_TARGET_SOURCE="external_mask",
        FAST_SEGMENTATION_LOADER=True,
        SEG_CACHE_MODE="none",
        SEG_CACHE_ROOT=str(tmp_path / "cache"),
        AUX_DETAIL_THRESHOLD=0.1,
        AUX_DETAIL_SCALES=[1, 2, 4],
        AUX_DETAIL_SUPPORT_KERNEL_SIZE=3,
    )

    train_dataset, val_dataset = build_train_val_datasets(data_config)

    assert isinstance(train_dataset, FastSegSupervisedDataset)
    assert isinstance(val_dataset, FastSegSupervisedDataset)
    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
