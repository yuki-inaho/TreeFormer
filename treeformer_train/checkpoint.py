from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .ema import ModelEma, unwrap_model


@dataclass
class CheckpointResult:
    saved_last: Path | None
    saved_best: Path | None
    saved_periodic: Path | None
    best_metric: float | None
    best_epoch: int | None


class CheckpointManager:
    """Rank-local checkpoint manager for last, best and periodic snapshots."""

    def __init__(
        self,
        save_dir: str | Path,
        *,
        metric_name: str = "val/smd",
        mode: str = "min",
        save_last: bool = True,
        save_best: bool = True,
        save_every: int = 0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"checkpoint mode must be 'min' or 'max', got {mode!r}")
        self.save_dir = Path(save_dir)
        self.metric_name = metric_name
        self.mode = mode
        self.save_last = save_last
        self.save_best = save_best
        self.save_every = int(save_every)
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        return metric < self.best_metric if self.mode == "min" else metric > self.best_metric

    def _payload(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        metrics: dict[str, float],
        ema: ModelEma | None,
        config: dict[str, Any] | None,
        extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else {},
            "metrics": metrics,
            "ema": ema.state_dict() if ema is not None else None,
            "config": config,
            "extra": extra or {},
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def save(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        metrics: dict[str, float],
        ema: ModelEma | None = None,
        config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CheckpointResult:
        saved_last: Path | None = None
        saved_best: Path | None = None
        saved_periodic: Path | None = None
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            raise KeyError(f"metrics must contain checkpoint metric {self.metric_name!r}; got {sorted(metrics)}")
        metric_value = float(metric_value)

        if self.is_better(metric_value):
            self.best_metric = metric_value
            self.best_epoch = int(epoch)
            if self.save_best:
                saved_best = self.save_dir / "best.pt"
                self._atomic_save(
                    self._payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=metrics,
                        ema=ema,
                        config=config,
                        extra=extra,
                    ),
                    saved_best,
                )

        if self.save_last:
            saved_last = self.save_dir / "last.pt"
            self._atomic_save(
                self._payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=metrics,
                    ema=ema,
                    config=config,
                    extra=extra,
                ),
                saved_last,
            )

        if self.save_every > 0 and epoch % self.save_every == 0:
            saved_periodic = self.save_dir / f"epoch_{epoch:06d}.pt"
            self._atomic_save(
                self._payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=metrics,
                    ema=ema,
                    config=config,
                    extra=extra,
                ),
                saved_periodic,
            )

        return CheckpointResult(saved_last, saved_best, saved_periodic, self.best_metric, self.best_epoch)


def load_training_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    required = {"epoch", "model", "optimizer", "scheduler", "metrics"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint {path} is missing required keys: {sorted(missing)}")
    return checkpoint


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_pretrained_model_weights(
    model: torch.nn.Module,
    path: str | Path | None,
    *,
    key: str = "net",
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> None:
    if not path:
        return

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"pretrained checkpoint must be a mapping: {path}")

    state_dict = checkpoint.get(key)
    if state_dict is None:
        available = sorted(str(item) for item in checkpoint)
        raise KeyError(f"pretrained checkpoint {path} has no key {key!r}; available keys: {available}")
    if not isinstance(state_dict, dict):
        raise ValueError(f"pretrained checkpoint key {key!r} must contain a state_dict")

    unwrap_model(model).load_state_dict(_strip_module_prefix(state_dict), strict=strict)
