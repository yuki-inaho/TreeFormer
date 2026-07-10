from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "val/aux_total_loss",
    "val/seg_iou",
    "val/seg_dice_score",
    "val/masked_heatmap_mae",
    "val/heatmap_peak_contrast",
    "val/paf_masked_l1",
    "time/epoch_seconds",
)

MAXIMIZE_HINTS = ("iou", "dice", "contrast", "precision", "recall")


def _metric_mode(tag: str) -> str:
    tag_lower = tag.lower()
    return "max" if any(item in tag_lower for item in MAXIMIZE_HINTS) else "min"


def _event_dirs(root: Path) -> list[Path]:
    return sorted({event.parent for event in root.rglob("events.out.tfevents.*")})


def _read_scalars(run_dir: Path, tags: tuple[str, ...]) -> dict[str, float | int | str]:
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    run_name = run_dir.parent.name if run_dir.name == "tensorboard" else run_dir.name
    row: dict[str, float | int | str] = {"run": run_name, "event_dir": str(run_dir)}
    last_step = 0
    for tag in tags:
        if tag not in available:
            continue
        values = accumulator.Scalars(tag)
        if not values:
            continue
        last = values[-1]
        mode = _metric_mode(tag)
        best = max(values, key=lambda item: item.value) if mode == "max" else min(values, key=lambda item: item.value)
        key = tag.replace("/", "__")
        row[f"{key}__last"] = float(last.value)
        row[f"{key}__best"] = float(best.value)
        row[f"{key}__best_step"] = int(best.step)
        last_step = max(last_step, int(last.step))
    row["last_step"] = last_step
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TensorBoard scalar metrics from dense-aux ablation runs")
    parser.add_argument("--runs-root", required=True, help="Directory containing Hydra run directories")
    parser.add_argument("--output-csv", required=True, help="CSV file to write")
    parser.add_argument("--tag", action="append", dest="tags", help="Scalar tag to include; repeatable")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_csv = Path(args.output_csv)
    tags = tuple(args.tags or DEFAULT_TAGS)
    rows = [_read_scalars(run_dir, tags) for run_dir in _event_dirs(runs_root)]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
