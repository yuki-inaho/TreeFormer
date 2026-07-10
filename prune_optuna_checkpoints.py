"""Delete checkpoint files for low-ranked Optuna trials produced by tune_graph_optuna.py.

Each pilot trial writes <output_root>/training/runs/trial_NNN_<seed>/checkpoints/best.pt,
which can be several hundred MB. After a pilot run this script keeps the checkpoints for the
top-N trials by score plus a set of always-kept trial numbers (e.g. the baseline trial),
and deletes the *.pt checkpoint files for every other trial. TensorBoard event files, Hydra
configs, and logs are never touched. Dry-run is the default; pass --apply to actually delete.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialRecord:
    number: int
    value: float | None
    failure: str | None
    run_dir_name: str | None


def select_kept_trial_numbers(records: list[TrialRecord], *, keep_top: int, always_keep: tuple[int, ...]) -> set[int]:
    if keep_top < 0:
        raise ValueError(f"keep_top must be >= 0, got {keep_top}")
    rankable = [
        record
        for record in records
        if record.failure is None and record.value is not None and math.isfinite(record.value)
    ]
    rankable.sort(key=lambda record: record.value, reverse=True)
    kept = {record.number for record in rankable[:keep_top]}
    kept.update(always_keep)
    return kept


def _resolve_run_dir(runs_root: Path, record: TrialRecord) -> Path | None:
    if record.run_dir_name is not None:
        run_dir = runs_root / record.run_dir_name
        return run_dir if run_dir.is_dir() else None
    matches = sorted(runs_root.glob(f"trial_{record.number:03d}_*"))
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous run dir for trial {record.number}: {matches}")
    if not matches:
        return None
    return matches[0]


def plan_checkpoint_deletions(
    runs_root: Path, records: list[TrialRecord], kept: set[int]
) -> tuple[list[Path], list[Path]]:
    delete_paths: list[Path] = []
    keep_paths: list[Path] = []
    for record in records:
        run_dir = _resolve_run_dir(runs_root, record)
        if run_dir is None:
            print(f"run dir not found for trial {record.number} (skipped)")
            continue
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            continue
        pt_files = sorted(checkpoints_dir.glob("*.pt"))
        destination = keep_paths if record.number in kept else delete_paths
        destination.extend(pt_files)
    return delete_paths, keep_paths


def assert_within(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"{target} is not within {root}") from None


def apply_deletions(delete_paths: list[Path], root: Path) -> int:
    for path in delete_paths:
        assert_within(root, path)
    freed_bytes = 0
    for path in delete_paths:
        freed_bytes += path.stat().st_size
        path.unlink()
    return freed_bytes


def _load_trial_records(study_name: str, db_path: Path) -> list[TrialRecord]:
    import optuna

    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    return [
        TrialRecord(
            number=trial.number,
            value=trial.value,
            failure=trial.user_attrs.get("failure"),
            run_dir_name=trial.user_attrs.get("run_dir_name"),
        )
        for trial in study.trials
    ]


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune checkpoint files for non-top Optuna pilot trials, keeping TensorBoard logs and configs."
    )
    parser.add_argument(
        "--output-root", type=Path, required=True, help="directory containing optuna_study.db and training/runs/"
    )
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--keep-top", type=int, default=3)
    parser.add_argument("--always-keep", default="0", help="comma-separated trial numbers to always keep")
    parser.add_argument("--apply", action="store_true", default=False, help="actually delete files (default: dry-run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = args.output_root / "optuna_study.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Optuna study database not found: {db_path}")
    runs_root = args.output_root / "training" / "runs"
    always_keep = tuple(int(item) for item in args.always_keep.split(",") if item.strip())

    records = _load_trial_records(args.study_name, db_path)
    kept = select_kept_trial_numbers(records, keep_top=args.keep_top, always_keep=always_keep)
    delete_paths, keep_paths = plan_checkpoint_deletions(runs_root, records, kept)

    print(f"study: {args.study_name}  trials: {len(records)}  kept trial numbers: {sorted(kept)}")
    print(f"{'action':<8} path")
    for path in keep_paths:
        print(f"{'KEEP':<8} {path}")
    for path in delete_paths:
        print(f"{'DELETE':<8} {path}")

    delete_bytes = sum(path.stat().st_size for path in delete_paths if path.exists())
    print(f"{len(delete_paths)} file(s) planned for deletion, {_format_bytes(delete_bytes)} would be freed")

    if not args.apply:
        print("dry-run: no files were deleted. Pass --apply to delete.")
        return

    freed_bytes = apply_deletions(delete_paths, runs_root)
    print(f"deleted {len(delete_paths)} file(s), freed {_format_bytes(freed_bytes)}")


if __name__ == "__main__":
    main()
