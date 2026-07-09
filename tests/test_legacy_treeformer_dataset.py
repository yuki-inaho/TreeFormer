from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from train_mst import LoadCNNDataset, custom_collate_fn


def _write_legacy_treeformer_sample(root, *, with_unet=True):
    (root / "data").mkdir(parents=True)
    (root / "img").mkdir()
    if with_unet:
        (root / "seg").mkdir()

    torch.save(
        SimpleNamespace(
            list_DETR_points_left_up=torch.tensor([[0.15, 0.15], [0.85, 0.85]], dtype=torch.float32),
            DETR_node_collections=torch.tensor([[0, 1]], dtype=torch.long),
        ),
        root / "data" / "sample.pt",
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    Image.fromarray(image).save(root / "img" / "sample.png")

    if with_unet:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:56, 10:54] = 255
        Image.fromarray(mask).save(root / "seg" / "sample.png")


def _write_virtual_root_treeformer_sample(root):
    (root / "data").mkdir(parents=True)
    (root / "img").mkdir()
    torch.save(
        SimpleNamespace(
            list_DETR_points_left_up=torch.tensor(
                [[0.1, 0.1], [0.2, 0.1], [0.8, 0.1], [0.9, 0.1]],
                dtype=torch.float32,
            ),
            DETR_node_collections=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            component_id=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            component_count=2,
            root_node_indices=torch.tensor([0, 2], dtype=torch.long),
            graph_topology="virtual_root_forest_v1",
        ),
        root / "data" / "sample.pt",
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    Image.fromarray(image).save(root / "img" / "sample.png")


def test_legacy_dataset_uses_external_seg_mask_when_requested(tmp_path):
    _write_legacy_treeformer_sample(tmp_path, with_unet=True)
    dataset = LoadCNNDataset(
        parent_path=tmp_path,
        max_size=64,
        is_train=False,
        is_rotate=False,
        segmentation_target_source="external_mask",
    )

    _, _, _, _, _, _, unet, _, _ = dataset[0]

    assert unet.dtype == torch.float32
    assert unet.shape == (32, 32)
    assert torch.equal(torch.unique(unet), torch.tensor([0.0, 1.0]))
    assert unet.float().mean() > 0.4
    assert unet[8, 8] == 1.0
    assert unet[0, 0] == 0.0


def test_legacy_dataset_requires_external_mask_when_configured(tmp_path):
    _write_legacy_treeformer_sample(tmp_path, with_unet=False)
    dataset = LoadCNNDataset(
        parent_path=tmp_path,
        max_size=64,
        is_train=False,
        is_rotate=False,
        segmentation_target_source="external_mask",
    )

    with pytest.raises(FileNotFoundError, match="external segmentation mask"):
        dataset[0]


def test_virtual_root_metadata_loads_from_pt(tmp_path):
    _write_virtual_root_treeformer_sample(tmp_path)
    dataset = LoadCNNDataset(
        parent_path=tmp_path,
        max_size=64,
        is_train=False,
        is_rotate=False,
        return_forest_metadata=True,
        strict_virtual_root_metadata=True,
    )

    sample = dataset[0]

    assert len(sample) == 10
    metadata = sample[-2]
    assert metadata["component_count"] == 2
    assert metadata["graph_topology"] == "virtual_root_forest_v1"
    assert torch.equal(metadata["component_id"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(metadata["root_node_indices"], torch.tensor([0, 2]))


def test_custom_collate_preserves_virtual_root_metadata(tmp_path):
    _write_virtual_root_treeformer_sample(tmp_path)
    dataset = LoadCNNDataset(
        parent_path=tmp_path,
        max_size=64,
        is_train=False,
        is_rotate=False,
        return_forest_metadata=True,
        strict_virtual_root_metadata=True,
    )

    batch = custom_collate_fn([dataset[0]])[0]

    assert isinstance(batch[-2], dict)
    assert batch[-1] == ["sample.pt"]
    assert torch.equal(batch[-2]["component_id"][0], torch.tensor([0, 0, 1, 1]))
    assert batch[-2]["component_count"] == [2]
