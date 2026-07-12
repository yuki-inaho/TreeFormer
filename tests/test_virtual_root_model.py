from types import SimpleNamespace
import sys
import types

import torch

sys.modules.setdefault(
    "MultiScaleDeformableAttention",
    types.SimpleNamespace(ms_deform_attn_forward=None, ms_deform_attn_backward=None),
)


class DummyNestedTensor:
    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor):
        self.tensor = tensor
        self.mask = mask

    def decompose(self):
        return self.tensor, self.mask


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.num_channels = [4]
        self.hidden_dim = hidden_dim

    def forward(self, samples):
        batch, _channels, height, width = samples.tensors.shape
        feature = torch.zeros((batch, 4, max(1, height // 8), max(1, width // 8)), dtype=samples.tensors.dtype)
        mask = torch.zeros((batch, feature.shape[-2], feature.shape[-1]), dtype=torch.bool)
        pos = torch.zeros((batch, self.hidden_dim, feature.shape[-2], feature.shape[-1]), dtype=samples.tensors.dtype)
        return [DummyNestedTensor(feature, mask)], [pos]


class DummyEncoderWithStride4(DummyEncoder):
    aux_num_channels = 6

    def forward(self, samples):
        features, pos = super().forward(samples)
        batch, _channels, height, width = samples.tensors.shape
        high = torch.zeros((batch, self.aux_num_channels, max(1, height // 4), max(1, width // 4)))
        high_mask = torch.zeros((batch, high.shape[-2], high.shape[-1]), dtype=torch.bool)
        return features, pos, DummyNestedTensor(high, high_mask)


class DummyDecoder(torch.nn.Module):
    def __init__(self, num_queries: int, hidden_dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def forward(self, srcs, masks, query_embeds, pos):
        batch = srcs[0].shape[0]
        hs = torch.zeros((batch, self.num_queries, self.hidden_dim), dtype=srcs[0].dtype)
        return hs, None, None, None, None


def _config(root_head_enabled: bool):
    hidden_dim = 32
    obj_token = 4
    rln_token = 1
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            NUM_CLASSES=2,
            GRAPH_OUTPUT_ENABLED=True,
            ROOT_HEAD=SimpleNamespace(ENABLED=root_head_enabled, HIDDEN_DIM=hidden_dim, NUM_LAYERS=2),
            AUX_HEAD=SimpleNamespace(ENABLED=False),
            DECODER=SimpleNamespace(
                OBJ_TOKEN=obj_token,
                RLN_TOKEN=rln_token,
                DUMMY_TOKEN=0,
                HIDDEN_DIM=hidden_dim,
                NUM_FEATURE_LEVELS=1,
                TWO_STAGE=False,
                AUX_LOSS=False,
                WITH_BOX_REFINE=False,
            ),
        )
    )


def _args():
    return SimpleNamespace(use_mst_train=True, use_gnn=False)


def test_relationformer_outputs_root_logits_when_enabled():
    from models.relationformer_2D import RelationFormer

    cfg = _config(root_head_enabled=True)
    model = RelationFormer(
        DummyEncoder(cfg.MODEL.DECODER.HIDDEN_DIM),
        DummyDecoder(cfg.MODEL.DECODER.OBJ_TOKEN + cfg.MODEL.DECODER.RLN_TOKEN, cfg.MODEL.DECODER.HIDDEN_DIM),
        cfg,
        _args(),
    )

    _h, out = model([torch.zeros((3, 32, 32), dtype=torch.float32)])

    assert out["pred_root_logits"].shape == (1, cfg.MODEL.DECODER.OBJ_TOKEN)


def test_relationformer_omits_root_logits_by_default():
    from models.relationformer_2D import RelationFormer

    cfg = _config(root_head_enabled=False)
    model = RelationFormer(
        DummyEncoder(cfg.MODEL.DECODER.HIDDEN_DIM),
        DummyDecoder(cfg.MODEL.DECODER.OBJ_TOKEN + cfg.MODEL.DECODER.RLN_TOKEN, cfg.MODEL.DECODER.HIDDEN_DIM),
        cfg,
        _args(),
    )

    _h, out = model([torch.zeros((3, 32, 32), dtype=torch.float32)])

    assert "pred_root_logits" not in out


def test_relationformer_can_condition_graph_features_on_aux_trunk():
    from models.relationformer_2D import RelationFormer

    cfg = _config(root_head_enabled=False)
    cfg.MODEL.AUX_HEAD = SimpleNamespace(
        ENABLED=True,
        HIDDEN_DIM=8,
        OUT_CHANNELS=5,
        GRAPH_CONDITIONING="aux_feature",
    )
    model = RelationFormer(
        DummyEncoder(cfg.MODEL.DECODER.HIDDEN_DIM),
        DummyDecoder(cfg.MODEL.DECODER.OBJ_TOKEN + cfg.MODEL.DECODER.RLN_TOKEN, cfg.MODEL.DECODER.HIDDEN_DIM),
        cfg,
        _args(),
    )

    _h, out = model([torch.zeros((3, 32, 32), dtype=torch.float32)])

    assert model.aux_graph_conditioning is not None
    assert out["aux_maps"].shape[1] == 5
    assert out["aux_maps"].shape[-2:] == (32, 32)
    assert out["aux_heatmap_native"].shape == (1, 1, 8, 8)
    assert out["aux_heatmap_offset_native"].shape == (1, 2, 8, 8)
    assert isinstance(model.aux_head.heatmap_head, torch.nn.Conv2d)


def test_relationformer_fuses_real_stride4_aux_feature_when_encoder_provides_it():
    from models.relationformer_2D import RelationFormer

    cfg = _config(root_head_enabled=False)
    cfg.MODEL.AUX_HEAD = SimpleNamespace(
        ENABLED=True,
        HIDDEN_DIM=8,
        OUT_CHANNELS=4,
        GRAPH_CONDITIONING="none",
    )
    model = RelationFormer(
        DummyEncoderWithStride4(cfg.MODEL.DECODER.HIDDEN_DIM),
        DummyDecoder(cfg.MODEL.DECODER.OBJ_TOKEN + cfg.MODEL.DECODER.RLN_TOKEN, cfg.MODEL.DECODER.HIDDEN_DIM),
        cfg,
        _args(),
    )

    _h, out = model([torch.zeros((3, 32, 32), dtype=torch.float32)])

    assert model.aux_head.high_resolution_encoder is not None
    assert model.aux_head.fusion_refine is not None
    assert out["aux_heatmap_native"].shape == (1, 1, 8, 8)
