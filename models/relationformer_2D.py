# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RelationFormer model and criterion classes.
"""

import torch
import torch.nn.functional as F
from torch import nn

# from torchvision.ops import nms
import copy

from .deformable_detr_backbone import build_backbone
from .deformable_detr_2D import build_deforamble_transformer
from .utils import nested_tensor_from_tensor_list, NestedTensor
########################################################################################################################


def _get_attr(config, name, default=None):
    return getattr(config, name, default)


def _group_count(channels, maximum=8):
    groups = min(int(maximum), int(channels))
    while int(channels) % groups != 0:
        groups -= 1
    return groups


def _conv_norm_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(_group_count(out_channels), out_channels),
        nn.ReLU(inplace=True),
    )


class AuxMapHead(nn.Module):
    """Stride-4 dense decoder with a direct native-grid node heatmap head.

    ``feature`` is the first TreeFormer feature level (stride 8).  The low
    resolution encoder refinement is kept separate because graph conditioning
    consumes it at stride 8.  The dense decoder then upsamples it to stride 4.
    Heatmap logits branch directly from that decoder feature through a 1x1
    projection; segmentation and direction are allowed their own lightweight
    output towers.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, decoder_stride=4):
        super().__init__()
        hidden_channels = int(hidden_channels)
        out_channels = int(out_channels)
        decoder_stride = int(decoder_stride)
        if hidden_channels <= 0:
            raise ValueError(f"aux head hidden_channels must be positive, got {hidden_channels}")
        if out_channels not in {4, 5}:
            raise ValueError(f"aux head out_channels must be 4 or 5, got {out_channels}")
        if decoder_stride not in {4, 8}:
            raise ValueError(f"aux head decoder_stride must be 4 or 8, got {decoder_stride}")

        self.decoder_stride = decoder_stride
        self.low_resolution_encoder = nn.Sequential(
            _conv_norm_relu(in_channels, hidden_channels),
            _conv_norm_relu(hidden_channels, hidden_channels),
        )
        self.decoder_refine = _conv_norm_relu(hidden_channels, hidden_channels)
        self.segmentation_tower = _conv_norm_relu(hidden_channels, hidden_channels)
        self.direction_tower = _conv_norm_relu(hidden_channels, hidden_channels)
        self.detail_tower = _conv_norm_relu(hidden_channels, hidden_channels) if out_channels == 5 else None

        self.segmentation_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # Intentionally direct: no heatmap-specific tower before this projection.
        self.heatmap_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.heatmap_offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.direction_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.detail_head = nn.Conv2d(hidden_channels, 1, kernel_size=1) if out_channels == 5 else None

    def forward(self, feature, output_size):
        _low_feature, logits, _heatmap_native, _offset_native = self.forward_with_features(feature, output_size)
        return logits

    def forward_with_features(self, feature, output_size):
        low_feature = self.low_resolution_encoder(feature)
        decoder_feature = low_feature
        if self.decoder_stride == 4:
            decoder_feature = F.interpolate(decoder_feature, scale_factor=2.0, mode="bilinear", align_corners=False)
        decoder_feature = self.decoder_refine(decoder_feature)

        segmentation_logits = self.segmentation_head(self.segmentation_tower(decoder_feature))
        heatmap_native = self.heatmap_head(decoder_feature)
        heatmap_offset_native = self.heatmap_offset_head(decoder_feature)
        direction_logits = self.direction_head(self.direction_tower(decoder_feature))
        native_maps = [segmentation_logits, heatmap_native, direction_logits]
        if self.detail_head is not None and self.detail_tower is not None:
            native_maps.append(self.detail_head(self.detail_tower(decoder_feature)))
        logits = torch.cat(native_maps, dim=1)
        full_resolution_logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return low_feature, full_resolution_logits, heatmap_native, heatmap_offset_native


class RelationFormer(nn.Module):
    """This is the RelationFormer module that performs object detection"""

    def __init__(self, encoder, decoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.use_mst_train = args.use_mst_train
        self.use_gnn = args.use_gnn

        self.num_queries = (
            config.MODEL.DECODER.OBJ_TOKEN + config.MODEL.DECODER.RLN_TOKEN + config.MODEL.DECODER.DUMMY_TOKEN
        )
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.hidden_dim = config.MODEL.DECODER.HIDDEN_DIM

        self.num_feature_levels = config.MODEL.DECODER.NUM_FEATURE_LEVELS
        self.two_stage = config.MODEL.DECODER.TWO_STAGE
        self.aux_loss = config.MODEL.DECODER.AUX_LOSS
        self.with_box_refine = config.MODEL.DECODER.WITH_BOX_REFINE
        self.num_classes = config.MODEL.NUM_CLASSES
        self.graph_output_enabled = bool(_get_attr(config.MODEL, "GRAPH_OUTPUT_ENABLED", True))
        root_head_config = _get_attr(config.MODEL, "ROOT_HEAD", None)
        self.root_head_enabled = root_head_config is not None and bool(_get_attr(root_head_config, "ENABLED", False))

        self.class_embed = nn.Linear(config.MODEL.DECODER.HIDDEN_DIM, 2)
        self.bbox_embed = MLP(config.MODEL.DECODER.HIDDEN_DIM, config.MODEL.DECODER.HIDDEN_DIM, 4, 3)
        self.root_embed = None
        if self.root_head_enabled:
            self.root_embed = MLP(
                config.MODEL.DECODER.HIDDEN_DIM,
                int(_get_attr(root_head_config, "HIDDEN_DIM", config.MODEL.DECODER.HIDDEN_DIM)),
                1,
                int(_get_attr(root_head_config, "NUM_LAYERS", 2)),
            )

        if config.MODEL.DECODER.RLN_TOKEN > 0:
            self.relation_embed = MLP(
                config.MODEL.DECODER.HIDDEN_DIM * (2 + config.MODEL.DECODER.RLN_TOKEN),
                config.MODEL.DECODER.HIDDEN_DIM,
                2,
                3,
            )
        else:
            self.relation_embed = MLP(
                config.MODEL.DECODER.HIDDEN_DIM * (2 + config.MODEL.DECODER.RLN_TOKEN),
                config.MODEL.DECODER.HIDDEN_DIM,
                2,
                3,
            )

        if not self.two_stage:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2)  # why *2
            # 因为后面做 torch.split，用来划分tensor，可以从数量上划分，还有维度上划分
            # query_embed, tgt = torch.split(query_embed, c, dim=1)
            # 其中c=self.hidden_dim 256
            # query_embed = 前半个 256 tgt为后半个256
        if self.num_feature_levels > 1:
            num_backbone_outs = len(self.encoder.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.encoder.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                ]
            )

        aux_head_config = _get_attr(config.MODEL, "AUX_HEAD", None)
        self.aux_head = None
        self.aux_graph_conditioning = None
        self.aux_graph_conditioning_mode = "none"
        if aux_head_config is not None and bool(_get_attr(aux_head_config, "ENABLED", False)):
            self.aux_graph_conditioning_mode = str(
                _get_attr(aux_head_config, "GRAPH_CONDITIONING", "none")
            ).lower().replace("-", "_")
            if self.aux_graph_conditioning_mode not in {"none", "aux_feature"}:
                raise ValueError(
                    "MODEL.AUX_HEAD.GRAPH_CONDITIONING must be one of ['none', 'aux_feature'], "
                    f"got {self.aux_graph_conditioning_mode!r}"
                )
            aux_hidden_dim = int(_get_attr(aux_head_config, "HIDDEN_DIM", max(self.hidden_dim // 4, 32)))
            self.aux_head = AuxMapHead(
                in_channels=self.hidden_dim,
                hidden_channels=aux_hidden_dim,
                out_channels=_get_attr(aux_head_config, "OUT_CHANNELS", 4),
                decoder_stride=_get_attr(aux_head_config, "DECODER_STRIDE", 4),
            )
            if self.aux_graph_conditioning_mode == "aux_feature":
                conditioning_groups = min(32, self.hidden_dim)
                while self.hidden_dim % conditioning_groups != 0:
                    conditioning_groups -= 1
                self.aux_graph_conditioning = nn.Sequential(
                    nn.Conv2d(aux_hidden_dim, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(conditioning_groups, self.hidden_dim),
                    nn.ReLU(inplace=True),
                )
        if not self.graph_output_enabled and self.aux_head is None:
            raise ValueError("MODEL.GRAPH_OUTPUT_ENABLED=false requires MODEL.AUX_HEAD.ENABLED=true")

        # self.decoder.decoder.bbox_embed = None

    def forward(self, samples):
        # 2*1*64*64
        # samples = nested_tensor_from_tensor_list([tensor.expand(3, -1, -1).contiguous() for tensor in samples])
        samples = nested_tensor_from_tensor_list(samples)
        # 2*3*64*64  # 不需要变成三倍

        # Deformable Transformer backbone
        features, pos = self.encoder(samples)
        # print(len(features))
        # 3
        # print(len(pos))
        # 3

        # Create
        srcs = []
        masks = []
        for level_idx, feat in enumerate(features):
            src, mask = feat.decompose()
            # print(src.shape)
            # torch.Size([2, 512, 8, 8])
            # print(mask) >>>就是是不是扩张的像素 由于大家都一样 所以不是
            # tensor([[[False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False]],
            #         [[False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False],
            #          [False, False, False, False, False, False, False, False]]],
            #        device='cuda:0')
            srcs.append(self.input_proj[level_idx](src))
            # self.input_proj = nn.ModuleList([
            #                 nn.Sequential(
            #                     nn.Conv2d(self.encoder.num_channels[0], self.hidden_dim, kernel_size=1),512*256
            #                     nn.GroupNorm(32, self.hidden_dim), 》》》 256
            #                     》》》torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
            #                 )])
            # print(srcs[0].shape)
            # torch.Size([2, 256, 8, 8])

            # 222222222
            # print(src.shape)
            # torch.Size([2, 1024, 4, 4])
            # 。。。
            # mask2*8*8 2*4*4 2*2*2 2*1*1
            # src2*256*8*8 2*256*4*4 2*256*2*2 2*256*1*1
            masks.append(mask)
            assert mask is not None

        out = {}
        if self.aux_head is not None:
            if self.aux_graph_conditioning is None:
                _aux_feature, aux_maps, heatmap_native, heatmap_offset_native = self.aux_head.forward_with_features(
                    srcs[0], samples.tensors.shape[-2:]
                )
                out["aux_maps"] = aux_maps
                out["aux_heatmap_native"] = heatmap_native
                out["aux_heatmap_offset_native"] = heatmap_offset_native
            else:
                aux_feature, aux_maps, heatmap_native, heatmap_offset_native = self.aux_head.forward_with_features(
                    srcs[0], samples.tensors.shape[-2:]
                )
                out["aux_maps"] = aux_maps
                out["aux_heatmap_native"] = heatmap_native
                out["aux_heatmap_offset_native"] = heatmap_offset_native
                srcs[0] = srcs[0] + self.aux_graph_conditioning(aux_feature)
        if not self.graph_output_enabled:
            return None, out

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)  # 3
            for level_idx in range(_len_srcs, self.num_feature_levels):
                if level_idx == _len_srcs:
                    src = self.input_proj[level_idx](features[-1].tensors)
                    # print(src.shape)
                    # torch.Size([2, 256, 1, 1])
                else:
                    src = self.input_proj[level_idx](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # print(mask)
                # tensor([[[False]],
                #         [[False]]], device='cuda:0') 2*64*64 》》 2*1*1src.shape[-2:]
                pos_l = self.encoder[1](NestedTensor(src, mask)).to(src.dtype)
                # print(pos_l.shape)
                # torch.Size([2, 256, 1, 1]) 创造了一个pos_sin
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        # 21*512
        # print(srcs[0].shape)
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        # torch.Size([2, 256, 1, 1])
        # print(masks[0].shape)
        # torch.Size([2, 8, 8])
        # torch.Size([2, 4, 4])
        # torch.Size([2, 2, 2])
        # torch.Size([2, 1, 1])
        # print(pos[0].shape)
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        # torch.Size([2, 256, 1, 1])

        hs, init_reference, inter_references, _, _ = self.decoder(srcs, masks, query_embeds, pos)
        # 2 21 256    2 21 2    2 21 2

        object_token = hs[..., : self.obj_token, :]
        # 2 20 256

        class_prob = self.class_embed(object_token)
        # 2 20 2

        coord_loc = self.bbox_embed(object_token).sigmoid()
        # 2 20 4

        out.update({"pred_logits": class_prob, "pred_nodes": coord_loc})
        if self.root_embed is not None:
            out["pred_root_logits"] = self.root_embed(object_token).squeeze(-1)
        return hs, out


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_relationformer(config, args, **kwargs):

    encoder = build_backbone(config)
    decoder = build_deforamble_transformer(config)

    model = RelationFormer(encoder, decoder, config, args, **kwargs)

    return model
