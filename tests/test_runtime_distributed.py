from __future__ import annotations

from types import SimpleNamespace

import torch.distributed as dist

from treeformer_train.runtime import setup_distributed


def _fake_process_group(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)


def test_setup_distributed_prefers_local_rank_env(monkeypatch):
    _fake_process_group(monkeypatch)
    monkeypatch.setenv("LOCAL_RANK", "1")

    context = setup_distributed(SimpleNamespace(mode="ddp", backend="nccl"))

    assert context.is_distributed is True
    assert context.rank == 1
    assert context.world_size == 2
    assert context.local_rank == 1


def test_setup_distributed_falls_back_to_current_device(monkeypatch):
    _fake_process_group(monkeypatch)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)

    context = setup_distributed(SimpleNamespace(mode="ddp", backend="nccl"))

    assert context.local_rank == 0


def test_setup_distributed_single_mode_has_no_process_group(monkeypatch):
    def _unexpected(*args, **kwargs):
        raise AssertionError("init_process_group must not be called for single mode")

    monkeypatch.setattr(dist, "init_process_group", _unexpected)

    context = setup_distributed(SimpleNamespace(mode="single"))

    assert context.is_distributed is False
    assert context.local_rank == 0
