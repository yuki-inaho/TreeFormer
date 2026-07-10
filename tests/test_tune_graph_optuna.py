import argparse
from pathlib import Path
from unittest.mock import patch

import optuna
import yaml

from tune_graph_optuna import (
    TrialExecutionError,
    TrialResult,
    baseline_trial_params,
    cache_root_for_heatmap_sigma,
    compute_graph_aux_score,
    heatmap_sigma_from_overrides,
    make_objective,
    parse_args,
    sanitize_failure_message,
    sanitize_overrides_for_report,
    suggest_overrides,
    verify_cache_root,
    _base_overrides,
    _find_run_dir,
    _write_reports,
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


def test_baseline_trial_params_match_the_shipped_joint_config():
    conf = yaml.safe_load(Path("conf/train/joint_virtual_root_aux.yaml").read_text())
    train = conf["TRAIN"]
    data = conf["DATA"]
    params = baseline_trial_params()

    # PyYAML's safe_load only recognizes exponent floats that contain a decimal
    # point (e.g. "1.0e-4"); bare "1e-4" is parsed as a plain string, so numeric
    # comparisons must cast explicitly rather than rely on YAML's implicit typing.
    assert params["lr"] == float(train["LR"])
    assert params["lr_backbone"] == float(train["LR_BACKBONE"])
    assert params["w_node"] == float(train["W_NODE"])
    assert params["w_edge"] == float(train["W_EDGE"])
    assert params["w_root"] == float(train["W_ROOT"])
    assert params["w_joint_aux"] == float(train["W_JOINT_AUX"])
    assert params["w_aux_seg"] == float(train["W_AUX_SEG"])
    assert params["w_aux_heatmap"] == float(train["W_AUX_HEATMAP"])
    assert params["w_aux_paf"] == float(train["W_AUX_PAF"])
    assert params["w_aux_detail"] == float(train["W_AUX_DETAIL"])
    assert params["clip_max_norm"] == float(train["CLIP_MAX_NORM"])

    assert float(data["AUX_HEATMAP_SIGMA"]) == 3.0
    assert float(train["W_AUX_HEATMAP_MSE"]) == 1.0
    assert float(train["W_AUX_HEATMAP_FOCAL"]) == 0.0
    assert float(train["W_AUX_HEATMAP_RIDGE"]) == 0.0
    assert params["heatmap_profile"] == "mse_baseline"


def test_baseline_params_are_reachable_from_the_search_space():
    fixed_trial = optuna.trial.FixedTrial(baseline_trial_params())

    suggest_overrides(fixed_trial)


def test_base_overrides_pin_the_data_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("TREEFORMER_PRIVATE_DATA", "/private/dataset/root")
    args = parse_args_with(["--data-seed", "1234"])

    overrides = _base_overrides(args, tmp_path, "trial_000", seg_cache_root=tmp_path / "cache")

    assert "DATA.SEED=1234" in overrides


def _make_run(runs_root: Path, name: str) -> Path:
    run_dir = runs_root / name / "checkpoints"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"")
    return run_dir.parent


def test_find_run_dir_selects_the_run_matching_the_pinned_data_seed(tmp_path):
    """conf/config.yaml:16 pins the run dir to `runs/{exp_name}_{DATA.SEED}`.

    A lexicographic `sorted(glob(...))[-1]` picks `trial_000_42` over
    `trial_000_3407`, because '4' > '3'. Resolve the directory exactly instead.
    """
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "trial_000_42")
    expected = _make_run(runs_root, "trial_000_3407")

    assert _find_run_dir(tmp_path, "trial_000", data_seed=3407) == expected


def test_find_run_dir_raises_when_the_seeded_run_has_no_checkpoint(tmp_path):
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "trial_000_99")

    try:
        _find_run_dir(tmp_path, "trial_000", data_seed=3407)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "trial_000_3407" in str(exc)


def test_verify_cache_root_reports_counts_for_a_populated_cache(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    for i in range(3):
        (train_dir / f"sample_{i}.pt").write_bytes(b"")
    for i in range(2):
        (val_dir / f"sample_{i}.pt").write_bytes(b"")

    counts = verify_cache_root(tmp_path)

    assert counts == {"train": 3, "val": 2}


def test_verify_cache_root_raises_when_cache_root_missing(tmp_path):
    missing = tmp_path / "does_not_exist"

    try:
        verify_cache_root(missing)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)


def test_verify_cache_root_raises_when_a_split_is_empty(tmp_path):
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    (train_dir / "sample_0.pt").write_bytes(b"")

    try:
        verify_cache_root(tmp_path)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "val" in str(exc)


def parse_args_with(argv: list[str]) -> argparse.Namespace:
    import sys

    old_argv = sys.argv
    sys.argv = ["tune_graph_optuna.py", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


def test_parse_args_defaults_pin_seeds_and_forbid_resume():
    args = parse_args_with([])

    assert args.sampler_seed == 3407
    assert args.data_seed == 3407
    assert args.allow_resume is False
    assert args.max_consecutive_failures == 3


def test_failed_trial_is_marked_fail_so_the_tpe_sampler_ignores_it(tmp_path, monkeypatch):
    """A crashed trial must not steer the search.

    optuna 4.9.0 builds its Parzen estimators from
    ``states = [TrialState.COMPLETE, TrialState.PRUNED]``
    (optuna/samplers/_tpe/sampler.py:599), and TPESampler has no
    ``consider_pruned_trials`` switch. Raising ``optuna.TrialPruned`` on an
    infrastructure failure would therefore feed that failure back into the
    surrogate model as a genuine "bad" observation. Only ``TrialState.FAIL``
    is invisible to the sampler.
    """
    monkeypatch.setenv("TREEFORMER_PRIVATE_DATA", "/private/dataset/root")
    args = parse_args_with(["--output-root", str(tmp_path / "out"), "--seg-cache-root", str(tmp_path / "cache")])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.sampler_seed))
    consecutive_failures = [0]
    objective = make_objective(args, study, tmp_path / "training", consecutive_failures)

    with (
        patch("tune_graph_optuna.subprocess.run", side_effect=RuntimeError("boom")),
        patch("tune_graph_optuna.verify_cache_root", return_value={"train": 1, "val": 1}),
    ):
        study.optimize(objective, n_trials=1, catch=(TrialExecutionError,))

    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.FAIL
    assert "boom" in trial.user_attrs["failure"]

    sampler_visible = study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
    )
    assert sampler_visible == []


def test_sanitize_failure_message_redacts_private_override_values():
    message = (
        "Command '['.venv/bin/python', 'train_hydra.py', 'DATA.DATA_PATH=/private/dataset/root', "
        "'checkpoint.pretrained=/private/checkpoint.pkl', 'DATA.BATCH_SIZE=2']' "
        "returned non-zero exit status 1."
    )

    sanitized = sanitize_failure_message(message)

    assert "/private/" not in sanitized
    assert "DATA.DATA_PATH=<redacted>" in sanitized
    assert "checkpoint.pretrained=<redacted>" in sanitized
    assert "DATA.BATCH_SIZE=2" in sanitized
    assert "returned non-zero exit status 1." in sanitized


def test_trial_failure_user_attr_never_stores_private_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TREEFORMER_PRIVATE_DATA", "/private/dataset/root")
    args = parse_args_with(["--output-root", str(tmp_path / "out"), "--seg-cache-root", str(tmp_path / "cache")])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.sampler_seed))
    objective = make_objective(args, study, tmp_path / "training", [0])
    failure = RuntimeError("Command '['train_hydra.py', 'DATA.DATA_PATH=/private/dataset/root']' failed")

    with (
        patch("tune_graph_optuna.subprocess.run", side_effect=failure),
        patch("tune_graph_optuna.verify_cache_root", return_value={"train": 1, "val": 1}),
    ):
        study.optimize(objective, n_trials=1, catch=(TrialExecutionError,))

    assert "/private/" not in study.trials[0].user_attrs["failure"]


def test_report_failure_section_never_contains_private_paths(tmp_path):
    failed = TrialResult(
        number=0,
        state="FAIL",
        value=None,
        params={"lr": 1e-4},
        metrics={},
        run_name="trial_000",
        failure="Command '['train_hydra.py', 'DATA.DATA_PATH=/private/dataset/root']' failed",
    )

    _write_reports(tmp_path, [failed])

    report = (tmp_path / "report.md").read_text()
    assert "/private/" not in report
    assert "DATA.DATA_PATH=<redacted>" in report


def test_consecutive_failures_abort_the_study(tmp_path, monkeypatch):
    monkeypatch.setenv("TREEFORMER_PRIVATE_DATA", "/private/dataset/root")
    args = parse_args_with(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--seg-cache-root",
            str(tmp_path / "cache"),
            "--max-consecutive-failures",
            "3",
        ]
    )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.sampler_seed))
    consecutive_failures = [0]
    objective = make_objective(args, study, tmp_path / "training", consecutive_failures)

    with (
        patch("tune_graph_optuna.subprocess.run", side_effect=RuntimeError("boom")),
        patch("tune_graph_optuna.verify_cache_root", return_value={"train": 1, "val": 1}),
    ):
        try:
            study.optimize(objective, n_trials=5, catch=(TrialExecutionError,))
            raise AssertionError("expected RuntimeError to abort the study")
        except RuntimeError as exc:
            assert not isinstance(exc, TrialExecutionError)
            assert "3" in str(exc)

    assert len(study.trials) == 3


def test_report_ranking_excludes_failed_trials(tmp_path):
    failed = TrialResult(
        number=0,
        state="COMPLETE",
        value=float("-inf"),
        params={"lr": 1e-4},
        metrics={},
        run_name="trial_000",
        failure="boom",
    )
    success_a = TrialResult(
        number=1,
        state="COMPLETE",
        value=0.5,
        params={"lr": 2e-4},
        metrics={
            "val/smd": 0.4,
            "val/seg_iou": 0.7,
            "val/heatmap_peak_contrast": 0.2,
            "val/paf_masked_l1": 0.2,
        },
        run_name="trial_001",
        failure=None,
    )
    success_b = TrialResult(
        number=2,
        state="COMPLETE",
        value=0.2,
        params={"lr": 3e-4},
        metrics={
            "val/smd": 0.5,
            "val/seg_iou": 0.6,
            "val/heatmap_peak_contrast": 0.1,
            "val/paf_masked_l1": 0.3,
        },
        run_name="trial_002",
        failure=None,
    )

    _write_reports(tmp_path, [failed, success_a, success_b])

    report = (tmp_path / "report.md").read_text()
    assert "-inf" not in report
    rank_one_rows = [line for line in report.splitlines() if line.startswith("| 1 |")]
    assert rank_one_rows, "expected a rank-1 row in the report"
    assert "trial_001" in rank_one_rows[0]
    assert "trial_000" not in report.split("## 失敗 trial")[0]
    assert "boom" in report


def test_best_trial_overrides_never_come_from_a_failed_trial(tmp_path):
    failed = TrialResult(
        number=0,
        state="COMPLETE",
        value=float("-inf"),
        params={"lr": 1e-4},
        metrics={},
        run_name="trial_000",
        failure="boom",
    )

    _write_reports(tmp_path, [failed])

    assert not (tmp_path / "best_trial_overrides.yaml").exists()

    success = TrialResult(
        number=1,
        state="COMPLETE",
        value=0.5,
        params={"lr": 9e-4},
        metrics={
            "val/smd": 0.4,
            "val/seg_iou": 0.7,
            "val/heatmap_peak_contrast": 0.2,
            "val/paf_masked_l1": 0.2,
        },
        run_name="trial_001",
        failure=None,
    )

    _write_reports(tmp_path, [failed, success])

    best_text = (tmp_path / "best_trial_overrides.yaml").read_text()
    assert str(success.params["lr"]) in best_text
