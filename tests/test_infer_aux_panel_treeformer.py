import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from infer_aux_panel_treeformer import (
    build_aux_panel_dataset,
    heatmap_to_pil,
    image_tensor_to_pil,
    make_aux_panel,
    make_detail_edge_map,
    mask_paf_by_segmentation,
    mask_scalar_map_by_segmentation,
    paf_to_rgb,
    overlay_node_peaks,
    resize_scalar_map,
    unpack_aux_panel_sample,
    write_aux_summary_json,
)


def test_make_detail_edge_map_finds_square_boundary():
    mask = torch.zeros(64, 64)
    mask[20:44, 22:42] = 1.0

    edge = make_detail_edge_map(mask)

    assert edge.shape == mask.shape
    assert edge.max() == 1.0
    assert edge.sum() > 0
    assert edge[20, 32] == 1.0
    assert edge[32, 32] == 0.0
    assert edge[0, 0] == 0.0


def test_aux_visualization_helpers_return_rgb_images():
    image = torch.zeros(3, 12, 10)
    heatmap = torch.rand(12, 10)
    paf = torch.zeros(2, 12, 10)
    paf[0, 3:8, 2:6] = 1.0

    assert image_tensor_to_pil(image).mode == "RGB"
    assert heatmap_to_pil(heatmap).mode == "RGB"
    assert paf_to_rgb(paf).mode == "RGB"
    assert overlay_node_peaks(Image.new("RGB", (10, 12)), torch.tensor([[4.0, 5.0, 0.9]])).mode == "RGB"


def test_resize_scalar_map_expands_native_heatmap_for_panel_display():
    native_heatmap = torch.zeros(2, 3)
    native_heatmap[1, 2] = 1.0

    display_heatmap = resize_scalar_map(native_heatmap, (8, 12))

    assert display_heatmap.shape == (8, 12)
    assert display_heatmap.max().item() > 0.9


def test_heatmap_visualization_is_black_outside_visible_mask():
    values = torch.ones(8, 10)
    visible = torch.zeros(8, 10)
    visible[2:6, 3:7] = 1.0

    image = np.asarray(heatmap_to_pil(values, visible_mask=visible))

    assert np.all(image[0, 0] == 0)
    assert np.any(image[3, 4] != 0)


def test_aux_visualization_helpers_mask_maps_by_segmentation_confidence():
    heatmap = torch.ones(8, 10)
    paf = torch.ones(2, 8, 10)
    empty_segmentation = torch.zeros(8, 10)
    foreground_segmentation = torch.zeros(8, 10)
    foreground_segmentation[2:6, 3:7] = 1.0

    assert mask_scalar_map_by_segmentation(heatmap, empty_segmentation).max().item() == 0.0
    assert mask_paf_by_segmentation(paf, empty_segmentation).max().item() == 0.0
    assert mask_scalar_map_by_segmentation(heatmap, foreground_segmentation).sum().item() == 16.0
    assert mask_paf_by_segmentation(paf, foreground_segmentation).sum().item() == 32.0


def test_make_aux_panel_and_summary_json(tmp_path: Path):
    image = torch.zeros(3, 16, 16)
    gt_segmentation = torch.zeros(16, 16)
    gt_segmentation[4:12, 4:12] = 1.0
    gt_heatmap = torch.zeros(16, 16)
    gt_heatmap[8, 8] = 1.0
    gt_paf = torch.zeros(16, 16, 2)
    gt_paf[4:12, 4:12, 0] = 1.0
    prediction = {
        "segmentation": gt_segmentation.clone(),
        "heatmap": gt_heatmap.clone(),
        "paf": gt_paf.permute(2, 0, 1).clone(),
    }

    panel = make_aux_panel(
        image=image,
        gt_segmentation=gt_segmentation,
        gt_heatmap=gt_heatmap,
        gt_paf=gt_paf,
        prediction=prediction,
        columns=3,
        panel_width=64,
        pad=4,
    )
    panel_path = tmp_path / "panel.png"
    panel.save(panel_path)

    json_path = tmp_path / "summary.json"
    write_aux_summary_json(
        json_path,
        sample_id="sample",
        prediction=prediction,
        targets={"segmentation": gt_segmentation, "heatmap": gt_heatmap, "paf": gt_paf.permute(2, 0, 1)},
    )

    assert panel_path.is_file()
    assert isinstance(panel, Image.Image)
    assert '"sample_id": "sample"' in json_path.read_text(encoding="utf-8")


def test_make_aux_panel_resizes_native_gt_heatmap_to_image_size():
    image = torch.zeros(3, 16, 16)
    segmentation = torch.zeros(16, 16)
    segmentation[4:12, 4:12] = 1.0
    native_heatmap = torch.zeros(4, 4)
    native_heatmap[2, 2] = 1.0
    paf = torch.zeros(16, 16, 2)
    prediction = {
        "segmentation": segmentation,
        "heatmap": torch.zeros(16, 16),
        "paf": torch.zeros(2, 16, 16),
    }

    panel = make_aux_panel(
        image=image,
        gt_segmentation=segmentation,
        gt_heatmap=native_heatmap,
        gt_paf=paf,
        prediction=prediction,
        columns=3,
        panel_width=64,
        pad=4,
    )

    assert panel.width > 0


def test_make_aux_panel_can_render_trained_detail_boundary(tmp_path: Path):
    image = torch.zeros(3, 16, 16)
    gt_segmentation = torch.zeros(16, 16)
    gt_segmentation[4:12, 4:12] = 1.0
    gt_heatmap = torch.zeros(16, 16)
    gt_paf = torch.zeros(16, 16, 2)
    prediction = {
        "segmentation": gt_segmentation.clone(),
        "heatmap": gt_heatmap.clone(),
        "paf": gt_paf.permute(2, 0, 1).clone(),
        "detail_boundary": make_detail_edge_map(gt_segmentation),
    }

    panel = make_aux_panel(
        image=image,
        gt_segmentation=gt_segmentation,
        gt_heatmap=gt_heatmap,
        gt_paf=gt_paf,
        prediction=prediction,
        columns=3,
        panel_width=64,
        pad=4,
    )
    panel.save(tmp_path / "detail_panel.png")
    json_path = tmp_path / "detail_summary.json"
    write_aux_summary_json(
        json_path,
        sample_id="sample",
        prediction=prediction,
        targets={"segmentation": gt_segmentation, "heatmap": gt_heatmap, "paf": gt_paf.permute(2, 0, 1)},
    )

    assert panel.width > 0
    assert "pred_detail_boundary_mean" in json_path.read_text(encoding="utf-8")


def test_build_aux_panel_dataset_uses_fast_loader_for_fast_seg_checkpoint(tmp_path: Path):
    split_root = tmp_path / "val"
    (split_root / "data").mkdir(parents=True)
    (split_root / "img").mkdir()
    (split_root / "seg").mkdir()
    torch.save(
        SimpleNamespace(
            list_DETR_points_left_up=torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.float32),
            DETR_node_collections=torch.tensor([[0, 1]], dtype=torch.long),
        ),
        split_root / "data" / "sample.pt",
    )
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    Image.fromarray(image).save(split_root / "img" / "sample.png")
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[8:32, 10:50] = 255
    Image.fromarray(mask).save(split_root / "seg" / "sample.png")
    checkpoint = {
        "config": {
            "DATA": {
                "FAST_SEGMENTATION_LOADER": True,
                "SEG_RESIZE_POLICY": "full",
                "AUX_TARGET_MODE": "seg_heatmap",
                "AUX_HEATMAP_SIGMA": 2.0,
            }
        }
    }

    dataset, loader_name = build_aux_panel_dataset(
        split_root=split_root,
        checkpoint=checkpoint,
        max_size=32,
        loader="auto",
    )
    image_tensor, _label, _pafs, segmentation, heatmap, _sample_id, sample_name = unpack_aux_panel_sample(dataset[0])

    assert loader_name == "fast-seg"
    assert sample_name == "sample"
    assert image_tensor.shape[-1] == 32
    assert segmentation.max().item() == 1.0
    assert heatmap.max().item() > 0.9


def test_unpack_aux_panel_sample_accepts_virtual_root_metadata():
    sample = (
        torch.zeros(3, 4, 5),
        "label",
        torch.zeros((2, 2)),
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((2, 4, 5)),
        torch.zeros((4, 5)),
        torch.zeros((4, 5)),
        torch.zeros((4, 5)),
        {
            "graph_topology": "virtual_root_forest_v1",
            "component_count": 2,
            "root_node_indices": torch.tensor([0, 2], dtype=torch.long),
            "root_edge_index": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            "component_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        },
        "sample.pt",
    )

    image, label, pafs, segmentation, heatmap, _sample_id, sample_name, metadata = unpack_aux_panel_sample(
        sample, return_metadata=True
    )

    assert image.shape == (3, 4, 5)
    assert label == "label"
    assert pafs.shape == (2, 4, 5)
    assert segmentation.shape == (4, 5)
    assert heatmap.shape == (4, 5)
    assert sample_name == "sample"
    assert metadata is not None
    assert metadata["graph_topology"] == "virtual_root_forest_v1"


def test_write_aux_summary_json_includes_virtual_root_metadata_when_present(tmp_path: Path):
    prediction = {
        "segmentation": torch.zeros(16, 16),
        "heatmap": torch.zeros(16, 16),
        "paf": torch.zeros(2, 16, 16),
    }
    forest_metadata = {
        "graph_topology": "virtual_root_forest_v1",
        "component_count": 2,
        "root_node_indices": torch.tensor([0, 2], dtype=torch.long),
        "root_edge_index": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        "component_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }

    output = tmp_path / "summary.json"
    write_aux_summary_json(
        output,
        sample_id="sample",
        prediction=prediction,
        targets={
            "segmentation": torch.zeros(16, 16),
            "heatmap": torch.zeros(16, 16),
            "paf": torch.zeros(2, 16, 16),
        },
        forest_metadata=forest_metadata,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "sample"
    assert payload["graph_topology"] == "virtual_root_forest_v1"
    assert payload["component_count"] == 2
    assert payload["root_node_indices"] == [0, 2]
    assert payload["root_edge_index"] == [[0, 1], [2, 3]]
    assert payload["component_id_summary"] == [
        {"component_id": 0, "node_count": 2},
        {"component_id": 1, "node_count": 2},
    ]
