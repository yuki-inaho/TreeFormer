from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


GRAPH_AUX_SCORE_WEIGHTS = {
    "val/smd": -1.0,
    "val/seg_iou": 0.10,
    "val/heatmap_peak_contrast": 0.05,
    "val/paf_masked_l1": -0.05,
}
REPORT_PRIVATE_KEYS = ("DATA.DATA_PATH", "DATA.TRAIN_PATH", "DATA.VAL_PATH", "checkpoint.pretrained")
_PRIVATE_VALUE_PATTERN = re.compile(
    "(" + "|".join(re.escape(key) for key in REPORT_PRIVATE_KEYS) + r")=[^'\"\s\]]+"
)


class TrialExecutionError(Exception):
    """A trial died for reasons unrelated to the hyperparameters it was given.

    Subprocess crashes, cache misses, and OOM say nothing about whether the
    sampled hyperparameters are good. Optuna 4.9.0 builds its TPE surrogate from
    ``states = [TrialState.COMPLETE, TrialState.PRUNED]``
    (optuna/samplers/_tpe/sampler.py:599) and offers no way to exclude pruned
    trials, so such failures must surface as ``TrialState.FAIL`` -- which the
    sampler never observes. Pass this type to ``study.optimize(catch=...)``.
    """


def sanitize_failure_message(message: str) -> str:
    """Redact private override values that subprocess errors echo back verbatim.

    ``CalledProcessError`` stringifies the whole argv, so the dataset root and the
    pretrained checkpoint path would otherwise land in the Optuna database and in
    report.md.
    """
    return _PRIVATE_VALUE_PATTERN.sub(r"\1=<redacted>", message)


@dataclass(frozen=True)
class TrialResult:
    number: int
    state: str
    value: float | None
    params: dict[str, Any]
    metrics: dict[str, float]
    run_name: str | None
    failure: str | None = None


def compute_graph_aux_score(metrics: dict[str, float]) -> float:
    missing = [key for key in GRAPH_AUX_SCORE_WEIGHTS if key not in metrics]
    if missing:
        raise KeyError(f"missing score metrics: {missing}")
    return sum(float(metrics[key]) * weight for key, weight in GRAPH_AUX_SCORE_WEIGHTS.items())


def sanitize_overrides_for_report(overrides: list[str]) -> list[str]:
    sanitized: list[str] = []
    for override in overrides:
        key = override.split("=", 1)[0]
        if key in REPORT_PRIVATE_KEYS:
            sanitized.append(f"{key}=<redacted>")
        else:
            sanitized.append(override)
    return sanitized


def _load_best_metrics(run_dir: Path) -> dict[str, float]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"checkpoint metrics must be a mapping: {checkpoint_path}")
    return {str(key): float(value) for key, value in metrics.items() if isinstance(value, int | float)}


def _find_run_dir(train_save_path: Path, run_name: str, *, data_seed: int) -> Path:
    """Resolve the run directory Hydra writes for this trial.

    conf/config.yaml:16 fixes it to ``runs/${log.exp_name}_${DATA.SEED}``, and
    _base_overrides pins both halves, so the directory is fully determined. Globbing
    ``{run_name}_*`` would additionally match runs left behind by a different
    ``--data-seed``, and picking ``sorted(...)[-1]`` off that glob compares seeds as
    strings ("_42" > "_3407").
    """
    run_dir = train_save_path / "runs" / f"{run_name}_{data_seed}"
    if not (run_dir / "checkpoints" / "best.pt").is_file():
        raise FileNotFoundError(f"no best checkpoint found for run {run_dir.name!r} under {train_save_path / 'runs'}")
    return run_dir


def suggest_overrides(trial: Any) -> list[str]:
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 3e-6, 1e-4, log=True)
    w_node = trial.suggest_categorical("w_node", [3.0, 5.0, 8.0])
    w_edge = trial.suggest_categorical("w_edge", [2.0, 4.0, 6.0])
    w_root = trial.suggest_categorical("w_root", [0.5, 1.0, 2.0])
    w_joint_aux = trial.suggest_categorical("w_joint_aux", [0.25, 0.5, 1.0])
    w_aux_seg = trial.suggest_categorical("w_aux_seg", [0.5, 1.0, 2.0])
    w_aux_heatmap = trial.suggest_categorical("w_aux_heatmap", [0.5, 1.0, 2.0])
    w_aux_paf = trial.suggest_categorical("w_aux_paf", [0.1, 0.25, 0.5])
    w_aux_detail = trial.suggest_categorical("w_aux_detail", [0.0, 0.05, 0.1])
    heatmap_profile = trial.suggest_categorical(
        "heatmap_profile",
        ["peak_focused", "peak_focused_sigma1_5", "focal_ridge", "mse_baseline"],
    )
    clip_max_norm = trial.suggest_categorical("clip_max_norm", [5.0, 10.0, 20.0])

    # Each profile pins sigma plus all six heatmap loss weights, so profiles are
    # self-contained and directly comparable across trials.
    heatmap_overrides = {
        "peak_focused": [
            "DATA.AUX_HEATMAP_SIGMA=3.0",
            "TRAIN.W_AUX_HEATMAP_MSE=0.25",
            "TRAIN.W_AUX_HEATMAP_FOCAL=1.0",
            "TRAIN.W_AUX_HEATMAP_RIDGE=0.05",
            "TRAIN.W_AUX_HEATMAP_COORD=0.05",
            "TRAIN.W_AUX_HEATMAP_COORD_VAR=0.001",
            "TRAIN.W_AUX_HEATMAP_PEAK=0.05",
        ],
        "peak_focused_sigma1_5": [
            "DATA.AUX_HEATMAP_SIGMA=1.5",
            "TRAIN.W_AUX_HEATMAP_MSE=0.25",
            "TRAIN.W_AUX_HEATMAP_FOCAL=1.0",
            "TRAIN.W_AUX_HEATMAP_RIDGE=0.05",
            "TRAIN.W_AUX_HEATMAP_COORD=0.05",
            "TRAIN.W_AUX_HEATMAP_COORD_VAR=0.001",
            "TRAIN.W_AUX_HEATMAP_PEAK=0.05",
        ],
        # Isolates the coord/var/peak contribution: same focal+ridge shape as
        # peak_focused, with the coord/peak terms switched off.
        "focal_ridge": [
            "DATA.AUX_HEATMAP_SIGMA=3.0",
            "TRAIN.W_AUX_HEATMAP_MSE=0.25",
            "TRAIN.W_AUX_HEATMAP_FOCAL=1.0",
            "TRAIN.W_AUX_HEATMAP_RIDGE=0.1",
            "TRAIN.W_AUX_HEATMAP_COORD=0.0",
            "TRAIN.W_AUX_HEATMAP_COORD_VAR=0.0",
            "TRAIN.W_AUX_HEATMAP_PEAK=0.0",
        ],
        # Pre-change behavior, kept as an ablation control.
        "mse_baseline": [
            "DATA.AUX_HEATMAP_SIGMA=3.0",
            "TRAIN.W_AUX_HEATMAP_MSE=1.0",
            "TRAIN.W_AUX_HEATMAP_FOCAL=0.0",
            "TRAIN.W_AUX_HEATMAP_RIDGE=0.0",
            "TRAIN.W_AUX_HEATMAP_COORD=0.0",
            "TRAIN.W_AUX_HEATMAP_COORD_VAR=0.0",
            "TRAIN.W_AUX_HEATMAP_PEAK=0.0",
        ],
    }[heatmap_profile]

    return [
        f"TRAIN.LR={lr}",
        f"TRAIN.LR_BACKBONE={lr_backbone}",
        f"TRAIN.W_NODE={w_node}",
        f"TRAIN.W_EDGE={w_edge}",
        f"TRAIN.W_ROOT={w_root}",
        f"TRAIN.W_JOINT_AUX={w_joint_aux}",
        f"TRAIN.W_AUX_SEG={w_aux_seg}",
        f"TRAIN.W_AUX_HEATMAP={w_aux_heatmap}",
        f"TRAIN.W_AUX_PAF={w_aux_paf}",
        f"TRAIN.W_AUX_DETAIL={w_aux_detail}",
        f"TRAIN.CLIP_MAX_NORM={clip_max_norm}",
        *heatmap_overrides,
    ]


def baseline_trial_params() -> dict[str, Any]:
    """Hyperparameters matching the shipped conf/train/joint_virtual_root_aux.yaml defaults.

    Keeping this in sync with the conf file is enforced by
    test_baseline_trial_params_match_the_shipped_joint_config.
    """
    return {
        "lr": 1e-4,
        "lr_backbone": 3e-5,
        "w_node": 5.0,
        "w_edge": 4.0,
        "w_root": 1.0,
        "w_joint_aux": 0.5,
        "w_aux_seg": 1.0,
        "w_aux_heatmap": 1.0,
        "w_aux_paf": 0.25,
        "w_aux_detail": 0.05,
        "heatmap_profile": "mse_baseline",
        "clip_max_norm": 20.0,
    }


def verify_cache_root(cache_root: Path, *, splits: tuple[str, ...] = ("train", "val")) -> dict[str, int]:
    if not cache_root.exists():
        raise FileNotFoundError(
            f"seg cache root not found: {cache_root}. "
            "Generate it with generate_fast_seg_cache.py for the matching heatmap sigma."
        )
    counts: dict[str, int] = {}
    for split in splits:
        split_dir = cache_root / split
        pt_count = len(list(split_dir.glob("*.pt"))) if split_dir.is_dir() else 0
        if not split_dir.is_dir() or pt_count == 0:
            raise FileNotFoundError(
                f"seg cache split {split!r} under {cache_root} is missing or empty. "
                "Generate it with generate_fast_seg_cache.py for the matching heatmap sigma."
            )
        counts[split] = pt_count
    return counts


def cache_root_for_heatmap_sigma(cache_root: Path, sigma: float) -> Path:
    normalized_sigma = f"{float(sigma):.4f}".rstrip("0").rstrip(".")
    if "." not in normalized_sigma:
        normalized_sigma += ".0"
    normalized_sigma = normalized_sigma.replace(".", "_")
    return cache_root / f"heatmap_sigma_{normalized_sigma}"


def heatmap_sigma_from_overrides(overrides: list[str]) -> float:
    prefix = "DATA.AUX_HEATMAP_SIGMA="
    for override in overrides:
        if override.startswith(prefix):
            return float(override.removeprefix(prefix))
    raise ValueError("Optuna trial overrides must set DATA.AUX_HEATMAP_SIGMA")


def _base_overrides(
    args: argparse.Namespace,
    train_save_path: Path,
    run_name: str,
    *,
    seg_cache_root: Path,
) -> list[str]:
    private_data = os.environ.get("TREEFORMER_PRIVATE_DATA", "")
    pretrained = os.environ.get(
        "TREEFORMER_PRETRAINED_CHECKPOINT",
        str(args.assets_root / "pretrained_weights/fork_source_main/grapevein/checkpoint_ours.pkl"),
    )
    if not private_data:
        raise RuntimeError("TREEFORMER_PRIVATE_DATA must be set for private joint graph+aux tuning")
    return [
        "train=joint_virtual_root_aux",
        "augmentation=disabled",
        "optimizer=muon_schedulefree",
        "ema=default",
        "ema.evaluate=false",
        "logging=tensorboard",
        f"checkpoint.pretrained={pretrained}",
        f"checkpoint.pretrained_key={os.environ.get('TREEFORMER_PRETRAINED_KEY', 'net')}",
        "checkpoint.pretrained_strict=false",
        "checkpoint.save_last=false",
        "checkpoint.save_best=true",
        "DATA.DATASET=treeformer-2D",
        f"DATA.DATA_PATH={private_data}",
        f"DATA.SEED={args.data_seed}",
        f"DATA.BATCH_SIZE={args.batch_size}",
        f"DATA.MAX_SIZE={args.max_size}",
        f"DATA.NUM_WORKERS={args.num_workers}",
        f"DATA.PERSISTENT_WORKERS={str(args.num_workers > 0).lower()}",
        "DATA.PREFETCH_FACTOR=2",
        f"DATA.SEG_CACHE_MODE={args.seg_cache_mode}",
        f"DATA.SEG_CACHE_ROOT={seg_cache_root}",
        "DATA.SEG_RESIZE_POLICY=full",
        f"DATA.TRAIN_LIMIT={args.train_limit}",
        f"DATA.VAL_LIMIT={args.val_limit}",
        f"TRAIN.SAVE_PATH={train_save_path}",
        f"TRAIN.EPOCHS={args.epochs}",
        "runtime.compile.aux_head=false",
        "runtime.compile.aux_loss=false",
        f"log.exp_name={run_name}",
    ]


def _write_reports(output_root: Path, results: list[TrialResult]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "trials.csv"
    fieldnames = sorted(
        {
            "number",
            "state",
            "value",
            "run_name",
            "failure",
            *{f"param/{key}" for result in results for key in result.params},
            *{f"metric/{key}" for result in results for key in result.metrics},
        }
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, Any] = {
                "number": result.number,
                "state": result.state,
                "value": result.value,
                "run_name": result.run_name,
                "failure": result.failure,
            }
            row.update({f"param/{key}": value for key, value in result.params.items()})
            row.update({f"metric/{key}": value for key, value in result.metrics.items()})
            writer.writerow(row)

    completed = [
        result
        for result in results
        if result.failure is None and result.value is not None and math.isfinite(result.value)
    ]
    completed.sort(key=lambda result: result.value if result.value is not None else float("-inf"), reverse=True)
    md_path = output_root / "report.md"
    with md_path.open("w") as handle:
        handle.write("# TreeFormer joint graph+aux Optuna 実験レポート\n\n")
        handle.write("このレポートは repo 外の実験結果を要約したものです。private dataset の実パスは記録しません。\n\n")
        handle.write("## スコア定義\n\n")
        handle.write(
            "最大化対象: `-val/smd + 0.10*val/seg_iou + 0.05*val/heatmap_peak_contrast - 0.05*val/paf_masked_l1`\n\n"
        )
        handle.write("## 上位 trial\n\n")
        handle.write("| rank | trial | score | val/smd | seg IoU | heatmap contrast | paf L1 | run |\n")
        handle.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for rank, result in enumerate(completed[:10], start=1):
            metrics = result.metrics
            handle.write(
                f"| {rank} | {result.number} | {result.value:.6f} | "
                f"{metrics.get('val/smd', float('nan')):.6f} | "
                f"{metrics.get('val/seg_iou', float('nan')):.6f} | "
                f"{metrics.get('val/heatmap_peak_contrast', float('nan')):.6f} | "
                f"{metrics.get('val/paf_masked_l1', float('nan')):.6f} | {result.run_name or ''} |\n"
            )
        handle.write("\n## 失敗 trial\n\n")
        failures = [result for result in results if result.failure]
        if not failures:
            handle.write("なし\n")
        else:
            for result in failures:
                handle.write(f"- trial {result.number}: {sanitize_failure_message(result.failure)}\n")
    if completed:
        best = completed[0]
        best_path = output_root / "best_trial_overrides.yaml"
        with best_path.open("w") as handle:
            handle.write("# Apply these values as Hydra overrides after reviewing the full report.\n")
            for key, value in sorted(best.params.items()):
                handle.write(f"{key}: {value}\n")


def _trial_results(study: Any) -> list[TrialResult]:
    results = []
    for trial in study.trials:
        results.append(
            TrialResult(
                number=trial.number,
                state=str(trial.state).removeprefix("TrialState."),
                value=trial.value,
                params=dict(trial.params),
                metrics=dict(trial.user_attrs.get("metrics", {})),
                run_name=trial.user_attrs.get("run_name"),
                failure=trial.user_attrs.get("failure"),
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for joint virtual-root graph + dense aux training")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", default="null")
    parser.add_argument("--val-limit", default="null")
    parser.add_argument("--study-name", default="joint_virtual_root_aux")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("TREEFORMER_ASSETS_ROOT", "../TreeFormer_assets"))
        / "optuna"
        / "joint_virtual_root_aux",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(os.environ.get("TREEFORMER_ASSETS_ROOT", "../TreeFormer_assets")),
    )
    parser.add_argument(
        "--seg-cache-root",
        type=Path,
        default=Path(
            os.environ.get("TREEFORMER_SEG_CACHE_ROOT", "../TreeFormer_assets/cache/fast_seg/private_seg_max640")
        ),
    )
    parser.add_argument("--seg-cache-mode", default=os.environ.get("TREEFORMER_SEG_CACHE_MODE", "disk"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--sampler-seed", type=int, default=3407)
    parser.add_argument("--data-seed", type=int, default=3407)
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        default=False,
        help="Explicitly resume an existing Optuna study database instead of failing.",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    return parser.parse_args()


def make_objective(
    args: argparse.Namespace,
    study: Any,
    train_save_path: Path,
    consecutive_failures: list[int],
) -> Callable[[Any], float]:
    def objective(trial: Any) -> float:
        run_name = f"trial_{trial.number:03d}"
        suggested_overrides = suggest_overrides(trial)
        cache_root = cache_root_for_heatmap_sigma(
            args.seg_cache_root,
            heatmap_sigma_from_overrides(suggested_overrides),
        )
        overrides = [
            *_base_overrides(args, train_save_path, run_name, seg_cache_root=cache_root),
            *suggested_overrides,
        ]
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("report_overrides", sanitize_overrides_for_report(overrides))
        trial.set_user_attr("data_seed", args.data_seed)
        trial.set_user_attr("sampler_seed", args.sampler_seed)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        command = [args.python, "train_hydra.py", *overrides]
        try:
            cache_counts = verify_cache_root(cache_root)
            trial.set_user_attr("cache_counts", cache_counts)
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent, env=env)
            run_dir = _find_run_dir(train_save_path, run_name, data_seed=args.data_seed)
            metrics = _load_best_metrics(run_dir)
            score = compute_graph_aux_score(metrics)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("run_dir_name", run_dir.name)
        except Exception as exc:
            failure = sanitize_failure_message(str(exc))
            trial.set_user_attr("failure", failure)
            _write_reports(args.output_root, _trial_results(study))
            consecutive_failures[0] += 1
            if consecutive_failures[0] >= args.max_consecutive_failures:
                raise RuntimeError(
                    f"aborting Optuna study after {consecutive_failures[0]} consecutive trial failures "
                    f"(--max-consecutive-failures={args.max_consecutive_failures}); last failure: {failure}"
                ) from exc
            raise TrialExecutionError(failure) from exc
        consecutive_failures[0] = 0
        _write_reports(args.output_root, _trial_results(study))
        return score

    return objective


def main() -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna tuning requires `uv pip install --python $TREEFORMER_PYTHON --project . --group tuning`."
        ) from exc

    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_save_path = args.output_root / "training"
    db_path = args.output_root / "optuna_study.db"
    if db_path.exists() and not args.allow_resume:
        raise RuntimeError(
            f"an Optuna study database already exists at {db_path}. "
            "Pass --allow-resume to explicitly resume it, or point --output-root at a new directory."
        )
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=args.allow_resume,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_trial_params())

    consecutive_failures = [0]
    objective = make_objective(args, study, train_save_path, consecutive_failures)

    study.optimize(objective, n_trials=args.trials, catch=(TrialExecutionError,))
    _write_reports(args.output_root, _trial_results(study))
    print(f"wrote Optuna report to {args.output_root / 'report.md'}")
    print(f"wrote Optuna trial CSV to {args.output_root / 'trials.csv'}")


if __name__ == "__main__":
    main()
