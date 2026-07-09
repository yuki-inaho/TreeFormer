from pathlib import Path

import torch
from PIL import Image

from infer_aux_panel_treeformer import (
    heatmap_to_pil,
    image_tensor_to_pil,
    make_aux_panel,
    make_detail_edge_map,
    paf_to_rgb,
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
