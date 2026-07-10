from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    mode: str
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def setup_reproducibility(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def setup_torch_performance(config: Any, device: torch.device) -> None:
    """Apply runtime performance flags that do not alter model structure."""

    deterministic = bool(_get(config, "deterministic", False))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif torch.are_deterministic_algorithms_enabled():
        torch.use_deterministic_algorithms(False)

    cuda_config = _get(config, "cuda", None)
    if device.type != "cuda" or cuda_config is None:
        return

    allow_tf32 = bool(_get(cuda_config, "allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    torch.backends.cudnn.benchmark = bool(_get(cuda_config, "cudnn_benchmark", False)) and not deterministic

    matmul_precision = _get(cuda_config, "float32_matmul_precision", None)
    if matmul_precision not in (None, ""):
        torch.set_float32_matmul_precision(str(matmul_precision))


def torch_compile_options(config: Any) -> dict[str, Any]:
    compile_config = _get(config, "compile", None)
    options: dict[str, Any] = {
        "mode": str(_get(compile_config, "mode", "reduce-overhead")),
        "fullgraph": bool(_get(compile_config, "fullgraph", True)),
        "dynamic": bool(_get(compile_config, "dynamic", False)),
    }
    backend = _get(compile_config, "backend", None)
    if backend not in (None, ""):
        options["backend"] = backend
    return options


def runtime_compile_enabled(config: Any, key: str) -> bool:
    compile_config = _get(config, "compile", None)
    return bool(_get(compile_config, key, False))


def amp_dtype(config: Any) -> torch.dtype:
    value = str(_get(_get(config, "amp", None), "dtype", "float16")).lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported runtime.amp.dtype: {value!r}")


def amp_context(device: torch.device, *, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_grad_scaler(*, enabled: bool, device: torch.device, dtype: torch.dtype) -> torch.amp.GradScaler | None:
    if not enabled or dtype is not torch.float16:
        return None
    return torch.amp.GradScaler(device=device.type, enabled=True)


def setup_distributed(config: Any) -> DistributedContext:
    mode = str(getattr(config, "mode", "single")) if not isinstance(config, dict) else str(config.get("mode", "single"))
    if mode == "single":
        return DistributedContext(mode="single", rank=0, world_size=1, local_rank=0, is_distributed=False)
    if mode != "ddp":
        raise ValueError(f"unsupported distributed.mode: {mode!r}")

    backend = getattr(config, "backend", "nccl") if not isinstance(config, dict) else config.get("backend", "nccl")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    return DistributedContext(
        mode="ddp",
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        is_distributed=True,
    )


def barrier(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.barrier()
