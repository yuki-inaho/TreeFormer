from pathlib import Path

import numpy as np
import torch
from PIL import Image

from infer_panel_treeformer import (
    discover_images,
    draw_graph_overlay,
    load_config_for_run,
    load_legacy_pt,
    make_panel_grid,
    parse_run_spec,
    save_graph_json,
    select_state_dict,
)


def test_parse_run_spec_accepts_hydra_checkpoint_without_config(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {}}, checkpoint)

    run = parse_run_spec(f"Ours|{checkpoint}|mst")

    assert run.label == "Ours"
    assert run.checkpoint == checkpoint
    assert run.mode == "mst"
    assert run.config is None
    assert run.use_mst is True
    assert run.use_distance is False


def test_parse_run_spec_accepts_explicit_legacy_config(tmp_path: Path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pkl"
    config.write_text("DATA: {}\nMODEL: {}\nTRAIN: {}\nlog: {}\n", encoding="utf-8")
    torch.save({"net": {}}, checkpoint)

    run = parse_run_spec(f"Legacy|{config}|{checkpoint}|mst-dist")

    assert run.label == "Legacy"
    assert run.config == config
    assert run.checkpoint == checkpoint
    assert run.use_mst is True
    assert run.use_distance is True


def test_select_state_dict_prefers_ema_shadow_and_strips_module_prefix():
    checkpoint = {
        "ema": {"shadow": {"module.linear.weight": torch.ones(1, 2)}},
        "model": {"linear.weight": torch.zeros(1, 2)},
    }

    state_dict = select_state_dict(checkpoint, weights="auto")

    assert list(state_dict) == ["linear.weight"]
    assert torch.equal(state_dict["linear.weight"], torch.ones(1, 2))


def test_select_state_dict_can_force_model_weights():
    checkpoint = {
        "ema": {"shadow": {"linear.weight": torch.ones(1, 2)}},
        "model": {"module.linear.weight": torch.zeros(1, 2)},
    }

    state_dict = select_state_dict(checkpoint, weights="model")

    assert torch.equal(state_dict["linear.weight"], torch.zeros(1, 2))


def test_load_config_for_run_uses_embedded_hydra_config():
    checkpoint = {
        "config": {
            "DATA": {"DATASET": "treeformer-2D"},
            "MODEL": {"DECODER": {"OBJ_TOKEN": 256, "RLN_TOKEN": 768}},
            "TRAIN": {"LR": 1e-4},
            "log": {"exp_name": "unit"},
        }
    }

    config = load_config_for_run(config_path=None, default_config_path=None, checkpoint=checkpoint)

    assert config.DATA.DATASET == "treeformer-2D"
    assert config.MODEL.DECODER.OBJ_TOKEN == 256
    assert config.log.exp_name == "unit"


def test_panel_rendering_and_graph_json_outputs(tmp_path: Path):
    image = Image.new("RGB", (64, 48), color=(90, 120, 150))
    nodes = np.array([[0.2, 0.2], [0.5, 0.6], [0.8, 0.4]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    overlay = draw_graph_overlay(image, nodes, edges, "Prediction", inset=True)
    panel = make_panel_grid([overlay], columns=1, pad=4, panel_width=80)
    panel_path = tmp_path / "sample_panel.png"
    panel.save(panel_path)

    graph_path = tmp_path / "sample_pred_graph.json"
    save_graph_json(graph_path, {"Prediction": (nodes, edges)})

    assert panel_path.is_file()
    assert panel.size[0] > 0
    assert '"Prediction"' in graph_path.read_text(encoding="utf-8")


def test_discover_images_sorts_supported_images(tmp_path: Path):
    (tmp_path / "b.txt").write_text("skip", encoding="utf-8")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")

    images = discover_images(tmp_path)

    assert [path.name for path in images] == ["a.jpg", "b.png"]


def test_load_legacy_pt_supports_mapping_payload(tmp_path: Path):
    path = tmp_path / "sample.pt"
    torch.save(
        {
            "list_DETR_points_left_up": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            "DETR_node_collections": torch.empty((0, 2), dtype=torch.long),
        },
        path,
    )

    parsed = load_legacy_pt(path)

    assert parsed is not None
    nodes, edges = parsed
    assert nodes.shape == (1, 2)
    assert edges.shape == (0, 2)
