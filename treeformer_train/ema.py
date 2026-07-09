from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEma:
    """Exponential moving average of model state with explicit apply/restore."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device: torch.device | str | None = None) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.device = torch.device(device) if device is not None else None
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] | None = None
        self._initialize(model)

    def _initialize(self, model: torch.nn.Module) -> None:
        state = unwrap_model(model).state_dict()
        self.shadow = {}
        for name, value in state.items():
            clone = value.detach().clone()
            if self.device is not None:
                clone = clone.to(self.device)
            self.shadow[name] = clone

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        state = unwrap_model(model).state_dict()
        for name, value in state.items():
            value = value.detach()
            target = self.shadow[name]
            if value.dtype.is_floating_point:
                target.mul_(self.decay).add_(value.to(target.device), alpha=1.0 - self.decay)
            else:
                target.copy_(value.to(target.device))

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        if self.backup is not None:
            raise RuntimeError("EMA weights are already applied; call restore() before applying again")
        module = unwrap_model(model)
        self.backup = {name: value.detach().clone() for name, value in module.state_dict().items()}
        module.load_state_dict({name: value.to(next(module.parameters()).device) if value.is_floating_point() else value for name, value in self.shadow.items()}, strict=True)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self.backup is None:
            raise RuntimeError("EMA restore() called before apply_to()")
        unwrap_model(model).load_state_dict(self.backup, strict=True)
        self.backup = None

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict.get("num_updates", 0))
        shadow = state_dict.get("shadow")
        if not isinstance(shadow, dict):
            raise ValueError("EMA state_dict must contain a shadow dictionary")
        self.shadow = {str(name): value.detach().clone() for name, value in shadow.items() if isinstance(value, torch.Tensor)}
