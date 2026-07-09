from types import SimpleNamespace

import torch

import epoch


class _DummyNet(torch.nn.Module):
    def forward(self, images):
        batch_size = len(images)
        hidden = torch.zeros(batch_size, 4, 8)
        output = {
            "pred_logits": torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
            "pred_nodes": torch.zeros(batch_size, 3, 4),
            "pred_root_logits": torch.zeros(batch_size, 3),
        }
        return hidden, output


class _DummySMD:
    def __call__(self, *, node_list, edge_list, pred_node_list, pred_edge_list):
        return torch.tensor([0.0])


def test_epoch_val_uses_virtual_root_mode_when_configured(monkeypatch):
    calls = []

    def fake_relation_infer(*args, **kwargs):
        calls.append(kwargs)
        return [torch.zeros(2, 2)], [torch.zeros(1, 2, dtype=torch.long).numpy()]

    monkeypatch.setattr(epoch, "shared_relation_infer", fake_relation_infer)
    config = SimpleNamespace(
        MODEL=SimpleNamespace(DECODER=SimpleNamespace(OBJ_TOKEN=3, RLN_TOKEN=1)),
        TRAIN=SimpleNamespace(POSTPROCESSOR_MODE="vr-mst", VR_ROOT_PENALTY=0.75),
    )
    args = SimpleNamespace(use_gnn=False, use_mst_train=True)
    batch = (
        [
            [torch.zeros(3, 8, 8)],
            [torch.zeros(2, 2)],
            [torch.zeros(1, 2, dtype=torch.long)],
        ],
    )

    result = epoch.epoch_val([batch], _DummyNet(), config, torch.device("cpu"), _DummySMD(), args)

    assert result == 0.0
    assert calls
    assert calls[0]["mode"] == "vr-mst"
    assert calls[0]["root_penalty"] == 0.75
