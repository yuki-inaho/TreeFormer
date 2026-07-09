import numpy as np
import torch

from inference_infinity_gradmst import relation_infer as unconstrained_infer
from inference_infinity_mst_nx_dist import relation_infer as distance_mst_infer
from inference_infinity_mst_nx_gradmst import relation_infer as mst_infer
from inference_treeformer import compute_mst_edges


class DummyRelationEmbed(torch.nn.Module):
    def __init__(self, costs):
        super().__init__()
        self.register_buffer("costs", torch.tensor(costs, dtype=torch.float32))

    def forward(self, features):
        costs = self.costs[: features.shape[0]].to(features.device)
        return torch.stack((torch.log(costs), torch.log1p(-costs)), dim=1)


class DummyModel(torch.nn.Module):
    def __init__(self, costs):
        super().__init__()
        self.relation_embed = DummyRelationEmbed(costs)


def make_inputs():
    h = torch.zeros((1, 4, 2), dtype=torch.float32)
    out = {
        "pred_logits": torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        "pred_nodes": torch.tensor([[[0.0, 0.0, 0.2, 0.2], [1.0, 0.0, 0.2, 0.2], [0.0, 1.0, 0.2, 0.2]]]),
    }
    return h, out


def test_unconstrained_infer_selects_edges_by_relation_argmax():
    h, out = make_inputs()
    model = DummyModel(costs=[0.1, 0.2, 0.9])

    nodes, edges = unconstrained_infer(h, out, model, obj_token=3, rln_token=1)

    assert nodes[0].shape == (3, 2)
    np.testing.assert_array_equal(edges[0], np.array([[0, 1], [0, 2]]))


def test_mst_infer_returns_tree_from_lowest_non_edge_costs():
    h, out = make_inputs()
    model = DummyModel(costs=[0.1, 0.2, 0.9])

    nodes, edges = mst_infer(h, out, model, obj_token=3, rln_token=1)

    assert nodes[0].shape == (3, 2)
    np.testing.assert_array_equal(edges[0], np.array([[0, 1], [0, 2]]))


def test_mst_helper_keeps_zero_cost_edges():
    node_pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long)
    costs = torch.tensor([0.0, 0.4, 0.6], dtype=torch.float32)

    edges = compute_mst_edges(node_pairs, costs)

    np.testing.assert_array_equal(edges, np.array([[0, 1], [0, 2]]))


def test_distance_mst_infer_accepts_optional_distance_weighting():
    h, out = make_inputs()
    model = DummyModel(costs=[0.9, 0.9, 0.1])

    _, edges = distance_mst_infer(
        h,
        out,
        model,
        obj_token=3,
        rln_token=1,
        use_distance=True,
        distance_weight=0.5,
    )

    assert edges[0].shape == (2, 2)
