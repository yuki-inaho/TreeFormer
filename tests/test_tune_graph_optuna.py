from pathlib import Path

from tune_graph_optuna import (
    cache_root_for_heatmap_sigma,
    compute_graph_aux_score,
    heatmap_sigma_from_overrides,
    parse_args,
    sanitize_overrides_for_report,
)


def test_compute_graph_aux_score_prefers_lower_smd_and_better_aux_metrics():
    baseline = {
        "val/smd": 0.5,
        "val/seg_iou": 0.6,
        "val/heatmap_peak_contrast": 0.1,
        "val/paf_masked_l1": 0.3,
    }
    better = {
        "val/smd": 0.4,
        "val/seg_iou": 0.7,
        "val/heatmap_peak_contrast": 0.2,
        "val/paf_masked_l1": 0.2,
    }

    assert compute_graph_aux_score(better) > compute_graph_aux_score(baseline)


def test_sanitize_overrides_for_report_redacts_private_inputs():
    overrides = [
        "DATA.DATA_PATH=/private/dataset/root",
        "DATA.BATCH_SIZE=8",
        "checkpoint.pretrained=/private/checkpoint.pkl",
        "TRAIN.EPOCHS=100",
    ]

    sanitized = sanitize_overrides_for_report(overrides)

    assert "DATA.DATA_PATH=<redacted>" in sanitized
    assert "checkpoint.pretrained=<redacted>" in sanitized
    assert "DATA.BATCH_SIZE=8" in sanitized
    assert all("/private/" not in item for item in sanitized)


def test_tune_module_does_not_require_optuna_for_report_helpers():
    assert Path("tune_graph_optuna.py").exists()


def test_optuna_defaults_use_the_profiled_safe_joint_runtime(monkeypatch):
    monkeypatch.setattr("sys.argv", ["tune_graph_optuna.py"])

    args = parse_args()

    assert args.batch_size == 2
    assert args.num_workers == 0


def test_optuna_selects_a_distinct_cache_root_for_each_heatmap_sigma():
    root = Path("cache")

    assert cache_root_for_heatmap_sigma(root, 3.0) == root / "heatmap_sigma_3_0"
    assert cache_root_for_heatmap_sigma(root, 1.5) == root / "heatmap_sigma_1_5"
    assert heatmap_sigma_from_overrides(["DATA.AUX_HEATMAP_SIGMA=1.5"]) == 1.5
