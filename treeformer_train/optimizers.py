from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import Optimizer


@dataclass(frozen=True)
class ParameterAssignment:
    name: str
    shape: tuple[int, ...]
    role: str
    lr: float
    weight_decay: float
    reason: str


@dataclass
class OptimizerBundle:
    optimizer: Any
    scheduler: Any
    assignments: list[ParameterAssignment]
    requires_train_eval: bool

    def write_parameter_report(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [assignment.__dict__ for assignment in self.assignments]
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class NullScheduler:
    """Scheduler no-op object with the same small contract used by training code."""

    def __init__(self) -> None:
        self.last_epoch = 0

    def step(self) -> None:
        self.last_epoch += 1

    def state_dict(self) -> dict[str, int]:
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_epoch = int(state_dict.get("last_epoch", 0))


def _orthogonalize_update(update: Tensor, ns_steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate orthogonalization used by Muon for matrix-like updates.

    The implementation intentionally avoids hidden fallbacks: tensors with fewer
    than two dimensions are not accepted by Muon parameter groups and therefore
    never reach this function.
    """

    if update.ndim < 2:
        raise ValueError("Muon update requires tensors with ndim >= 2")

    original_shape = update.shape
    matrix = update.reshape(update.shape[0], -1).float()
    transposed = False
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
        transposed = True

    matrix = matrix / (matrix.norm(dim=(0, 1), keepdim=True) + eps)
    # Coefficients commonly used by the Muon Newton-Schulz quintic iteration.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        gram = matrix @ matrix.T
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix

    if transposed:
        matrix = matrix.T
    return matrix.reshape(original_shape).to(dtype=update.dtype, device=update.device)


class MuonAdamW(Optimizer):
    """Single optimizer containing Muon groups and AdamW auxiliary groups.

    Muon is applied only to explicitly assigned hidden matrix-like parameters.
    Biases, normalization parameters, embeddings, heads, offsets and all tensors
    with fewer than two dimensions are handled by the AdamW auxiliary update.
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        if not param_groups:
            raise ValueError("MuonAdamW requires at least one parameter group")
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            role = group.get("role")
            if role == "muon":
                self._step_muon_group(group)
            elif role == "adamw_aux":
                self._step_adamw_group(group)
            else:
                raise ValueError(f"unsupported MuonAdamW group role: {role!r}")
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        momentum = float(group.get("momentum", 0.95))
        weight_decay = float(group.get("weight_decay", 0.0))
        ns_steps = int(group.get("ns_steps", 5))
        nesterov = bool(group.get("nesterov", True))

        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients")
            if param.ndim < 2:
                raise RuntimeError("Muon group received a tensor with ndim < 2; fix parameter partitioning")

            grad = param.grad.detach()
            state = self.state[param]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = torch.zeros_like(param)
                state["momentum_buffer"] = buffer
            buffer.mul_(momentum).add_(grad)
            update = grad.add(buffer, alpha=momentum) if nesterov else buffer
            update = _orthogonalize_update(update, ns_steps=ns_steps)

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update, alpha=-lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        weight_decay = float(group.get("weight_decay", 0.0))

        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("MuonAdamW AdamW auxiliary update does not support sparse gradients")

            grad = param.grad.detach()
            state = self.state[param]
            if "step" not in state:
                state["step"] = torch.tensor(0.0, device=param.device)
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            step = int(state["step"].item())
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)


def _plain(config: Any) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return dict(config)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    current = config
    for part in key.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def _iter_trainable_named_parameters(model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def _matches_any(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _legacy_lr_for_name(name: str, train_cfg: Any, lr: float, lr_backbone: float) -> float:
    lowered = name.lower()
    if "encoder.0" in lowered:
        return lr_backbone
    if "reference_points" in lowered or "sampling_offsets" in lowered:
        return lr * 0.1
    return lr


def _build_adamw_groups(model: torch.nn.Module, train_cfg: Any) -> tuple[list[dict[str, Any]], list[ParameterAssignment]]:
    lr = float(_get(train_cfg, "LR", 1e-4))
    lr_backbone = float(_get(train_cfg, "LR_BACKBONE", lr))
    weight_decay = float(_get(train_cfg, "WEIGHT_DECAY", 0.0))
    assignments: list[ParameterAssignment] = []
    buckets: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in _iter_trainable_named_parameters(model):
        param_lr = _legacy_lr_for_name(name, train_cfg, lr, lr_backbone)
        buckets.setdefault(param_lr, []).append(parameter)
        assignments.append(
            ParameterAssignment(name, tuple(parameter.shape), "adamw", param_lr, weight_decay, "legacy_lr_partition")
        )
    return ([{"params": params, "lr": group_lr, "weight_decay": weight_decay} for group_lr, params in buckets.items()], assignments)


def _split_muon_parameters(
    model: torch.nn.Module,
    train_cfg: Any,
    optimizer_cfg: Any,
) -> tuple[list[dict[str, Any]], list[ParameterAssignment]]:
    lr = float(_get(optimizer_cfg, "lr", _get(train_cfg, "LR", 1e-4)))
    lr_backbone = float(_get(optimizer_cfg, "lr_backbone", _get(train_cfg, "LR_BACKBONE", lr)))
    muon_weight_decay = float(_get(optimizer_cfg, "muon_weight_decay", _get(train_cfg, "WEIGHT_DECAY", 0.0)))
    aux_weight_decay = float(_get(optimizer_cfg, "aux_weight_decay", _get(train_cfg, "WEIGHT_DECAY", 0.0)))
    aux_keywords = list(
        _get(
            optimizer_cfg,
            "aux_keywords",
            ["bias", "norm", "bn", "ln", "embedding", "embed", "head", "reference_points", "sampling_offsets"],
        )
    )
    force_muon_keywords = list(_get(optimizer_cfg, "force_muon_keywords", []))

    muon_params: list[torch.nn.Parameter] = []
    aux_buckets: dict[float, list[torch.nn.Parameter]] = {}
    assignments: list[ParameterAssignment] = []

    for name, parameter in _iter_trainable_named_parameters(model):
        force_muon = _matches_any(name, force_muon_keywords)
        excluded = _matches_any(name, aux_keywords)
        if parameter.ndim >= 2 and (force_muon or not excluded):
            muon_params.append(parameter)
            assignments.append(ParameterAssignment(name, tuple(parameter.shape), "muon", lr, muon_weight_decay, "matrix_hidden_weight"))
        else:
            param_lr = _legacy_lr_for_name(name, train_cfg, lr, lr_backbone)
            aux_buckets.setdefault(param_lr, []).append(parameter)
            reason = "ndim_lt_2" if parameter.ndim < 2 else "aux_keyword_or_head"
            assignments.append(ParameterAssignment(name, tuple(parameter.shape), "adamw_aux", param_lr, aux_weight_decay, reason))

    if not muon_params:
        raise ValueError("muon_schedulefree selected but no trainable parameter was assigned to Muon")

    groups: list[dict[str, Any]] = [
        {
            "params": muon_params,
            "role": "muon",
            "lr": lr,
            "weight_decay": muon_weight_decay,
            "momentum": float(_get(optimizer_cfg, "muon_momentum", 0.95)),
            "nesterov": bool(_get(optimizer_cfg, "muon_nesterov", True)),
            "ns_steps": int(_get(optimizer_cfg, "muon_ns_steps", 5)),
        }
    ]
    for group_lr, params in aux_buckets.items():
        groups.append(
            {
                "params": params,
                "role": "adamw_aux",
                "lr": group_lr,
                "weight_decay": aux_weight_decay,
                "betas": tuple(_get(optimizer_cfg, "aux_betas", (0.9, 0.999))),
                "eps": float(_get(optimizer_cfg, "aux_eps", 1e-8)),
            }
        )
    return groups, assignments


def _import_schedulefree() -> Any:
    try:
        import schedulefree
    except ImportError as exc:
        raise ImportError(
            "optimizer.name requires the 'schedulefree' package. Install with `uv sync` after updating pyproject.toml."
        ) from exc
    return schedulefree


def build_optimizer_bundle(model: torch.nn.Module, train_cfg: Any, optimizer_cfg: Any) -> OptimizerBundle:
    name = str(_get(optimizer_cfg, "name", "adamw_step")).lower()

    if name == "adamw_step":
        groups, assignments = _build_adamw_groups(model, train_cfg)
        optimizer = torch.optim.AdamW(groups, lr=float(_get(train_cfg, "LR", 1e-4)), weight_decay=float(_get(train_cfg, "WEIGHT_DECAY", 0.0)))
        lr_drop = int(_get(optimizer_cfg, "lr_drop", _get(train_cfg, "LR_DROP", 100)))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop)
        return OptimizerBundle(optimizer=optimizer, scheduler=scheduler, assignments=assignments, requires_train_eval=False)

    if name == "schedulefree_adamw":
        schedulefree = _import_schedulefree()
        groups, assignments = _build_adamw_groups(model, train_cfg)
        optimizer = schedulefree.AdamWScheduleFree(
            groups,
            lr=float(_get(optimizer_cfg, "lr", _get(train_cfg, "LR", 1e-4))),
            betas=tuple(_get(optimizer_cfg, "betas", (0.9, 0.999))),
            eps=float(_get(optimizer_cfg, "eps", 1e-8)),
            weight_decay=float(_get(optimizer_cfg, "weight_decay", _get(train_cfg, "WEIGHT_DECAY", 0.0))),
            warmup_steps=int(_get(optimizer_cfg, "warmup_steps", 0)),
            r=float(_get(optimizer_cfg, "r", 0.0)),
            weight_lr_power=float(_get(optimizer_cfg, "weight_lr_power", 2.0)),
            foreach=bool(_get(optimizer_cfg, "foreach", True)),
        )
        return OptimizerBundle(optimizer=optimizer, scheduler=NullScheduler(), assignments=assignments, requires_train_eval=True)

    if name == "muon_schedulefree":
        schedulefree = _import_schedulefree()
        groups, assignments = _split_muon_parameters(model, train_cfg, optimizer_cfg)
        base_optimizer = MuonAdamW(groups)
        optimizer = schedulefree.ScheduleFreeWrapper(
            base_optimizer,
            weight_decay_at_y=float(_get(optimizer_cfg, "weight_decay_at_y", 0.0)),
            momentum=float(_get(optimizer_cfg, "outer_momentum", 0.9)),
            weight_lr_power=float(_get(optimizer_cfg, "weight_lr_power", 2.0)),
            r=float(_get(optimizer_cfg, "r", 0.0)),
        )
        return OptimizerBundle(optimizer=optimizer, scheduler=NullScheduler(), assignments=assignments, requires_train_eval=True)

    raise ValueError(f"unsupported optimizer.name: {name!r}")


def set_optimizer_train_mode(optimizer: Any, *, required: bool) -> None:
    if hasattr(optimizer, "train"):
        optimizer.train()
    elif required:
        raise TypeError("optimizer requires train/eval mode contract but has no train() method")


def set_optimizer_eval_mode(optimizer: Any, *, required: bool) -> None:
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    elif required:
        raise TypeError("optimizer requires train/eval mode contract but has no eval() method")
