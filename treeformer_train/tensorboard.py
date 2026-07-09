from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorBoardLogger:
    """Rank-aware TensorBoard scalar logger."""

    def __init__(self, log_dir: str | Path, *, enabled: bool = True, rank: int = 0, flush_secs: int = 30) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.writer: Any | None = None
        if self.enabled and self.rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError("TensorBoard logging requires the 'tensorboard' package. Run `uv sync`.") from exc
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def add_scalars(self, step: int, scalars: dict[str, float]) -> None:
        if self.writer is None:
            return
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, float(value), int(step))

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
