from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from treeformer_train.optimizers import build_optimizer_bundle, set_optimizer_eval_mode, set_optimizer_train_mode


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8))
        self.hidden = torch.nn.Linear(8, 4)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.hidden(self.encoder(x))).sum()


def _train_cfg():
    return SimpleNamespace(LR=1e-3, LR_BACKBONE=1e-4, WEIGHT_DECAY=0.01, LR_DROP=3)


def _run_one_step(bundle, model):
    set_optimizer_train_mode(bundle.optimizer, required=bundle.requires_train_eval)
    loss = model(torch.randn(2, 4))
    bundle.optimizer.zero_grad()
    loss.backward()
    bundle.optimizer.step()
    set_optimizer_eval_mode(bundle.optimizer, required=bundle.requires_train_eval)


def test_adamw_step_factory_builds_steplr_and_assigns_all_parameters():
    model = TinyModel()
    cfg = OmegaConf.create({"name": "adamw_step", "lr_drop": 3})
    bundle = build_optimizer_bundle(model, _train_cfg(), cfg)

    assert bundle.requires_train_eval is False
    assert bundle.scheduler.__class__.__name__ == "StepLR"
    assert len(bundle.assignments) == len([p for p in model.parameters() if p.requires_grad])
    _run_one_step(bundle, model)


def test_schedulefree_adamw_requires_explicit_mode_switch():
    model = TinyModel()
    cfg = OmegaConf.create({"name": "schedulefree_adamw", "lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 0})
    bundle = build_optimizer_bundle(model, _train_cfg(), cfg)

    assert bundle.requires_train_eval is True
    with pytest.raises(Exception, match="train mode"):
        loss = model(torch.randn(2, 4))
        bundle.optimizer.zero_grad()
        loss.backward()
        bundle.optimizer.step()
    _run_one_step(bundle, model)


def test_muon_schedulefree_partitions_matrix_weights_and_aux_parameters():
    model = TinyModel()
    cfg = OmegaConf.create(
        {
            "name": "muon_schedulefree",
            "lr": 1e-3,
            "lr_backbone": 1e-4,
            "muon_weight_decay": 0.01,
            "aux_weight_decay": 0.01,
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_ns_steps": 2,
            "aux_betas": [0.9, 0.999],
            "aux_eps": 1e-8,
            "outer_momentum": 0.9,
            "weight_decay_at_y": 0.0,
            "weight_lr_power": 2.0,
            "r": 0.0,
            "aux_keywords": ["bias", "norm", "head"],
            "force_muon_keywords": [],
        }
    )
    bundle = build_optimizer_bundle(model, _train_cfg(), cfg)

    roles = {assignment.name: assignment.role for assignment in bundle.assignments}
    assert roles["hidden.weight"] == "muon"
    assert roles["head.weight"] == "adamw_aux"
    assert roles["encoder.1.weight"] == "adamw_aux"
    assert bundle.requires_train_eval is True
    _run_one_step(bundle, model)
