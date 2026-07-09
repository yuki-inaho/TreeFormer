from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from treeformer_train.checkpoint import CheckpointManager, load_pretrained_model_weights
from treeformer_train.ema import ModelEma
from treeformer_train.optimizers import NullScheduler
from treeformer_train.tensorboard import TensorBoardLogger


def test_model_ema_update_apply_and_restore_changes_and_restores_weights():
    model = torch.nn.Linear(2, 1)
    ema = ModelEma(model, decay=0.5)
    original = model.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(2.0)
    changed = model.weight.detach().clone()
    ema.update(model)

    assert ema.num_updates == 1
    assert not torch.allclose(ema.shadow["weight"], original)
    assert not torch.allclose(ema.shadow["weight"], changed)

    ema.apply_to(model)
    assert torch.allclose(model.weight, ema.shadow["weight"])
    ema.restore(model)
    assert torch.allclose(model.weight, changed)


def test_checkpoint_manager_saves_last_and_best_only_on_metric_improvement(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = NullScheduler()
    manager = CheckpointManager(tmp_path, metric_name="val/smd", mode="min", save_last=True, save_best=True, save_every=2)

    first = manager.save(epoch=1, model=model, optimizer=optimizer, scheduler=scheduler, metrics={"val/smd": 0.5})
    assert first.saved_last == tmp_path / "last.pt"
    assert first.saved_best == tmp_path / "best.pt"
    assert first.best_metric == 0.5

    second = manager.save(epoch=2, model=model, optimizer=optimizer, scheduler=scheduler, metrics={"val/smd": 0.6})
    assert second.saved_last == tmp_path / "last.pt"
    assert second.saved_best is None
    assert second.saved_periodic == tmp_path / "epoch_000002.pt"
    assert manager.best_metric == 0.5

    third = manager.save(epoch=3, model=model, optimizer=optimizer, scheduler=scheduler, metrics={"val/smd": 0.4})
    assert third.saved_best == tmp_path / "best.pt"
    assert manager.best_metric == 0.4
    assert torch.load(tmp_path / "best.pt", map_location="cpu")["epoch"] == 3


def test_tensorboard_logger_writes_scalar_event_file(tmp_path: Path):
    logger = TensorBoardLogger(tmp_path, enabled=True, rank=0, flush_secs=1)
    logger.add_scalars(1, {"train/total_loss": 1.25, "val/smd": 0.5})
    logger.close()

    event_files = list(tmp_path.glob("events.out.tfevents.*"))
    assert event_files, "TensorBoard event file was not created"

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    assert "train/total_loss" in tags
    assert "val/smd" in tags


def test_load_pretrained_model_weights_reads_legacy_net_with_module_prefix(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    source = torch.nn.Linear(2, 1)
    checkpoint = {
        "net": {
            f"module.{name}": value.detach().clone()
            for name, value in source.state_dict().items()
        }
    }
    path = tmp_path / "legacy.pkl"
    torch.save(checkpoint, path)

    load_pretrained_model_weights(model, path)

    assert torch.allclose(model.weight, source.weight)
    assert torch.allclose(model.bias, source.bias)
