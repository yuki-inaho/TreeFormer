from __future__ import annotations

import random
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
