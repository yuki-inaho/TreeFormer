import numpy as np
import pytest
import torch

from inference_infinity_gradmst import relation_infer as unconstrained_infer
from inference_infinity_mst_nx_dist import relation_infer as distance_mst_infer
from inference_infinity_mst_nx_gradmst import relation_infer as mst_infer
from inference_treeformer import compute_mst_edges, compute_virtual_root_forest_edges, relation_infer


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


def test_relation_infer_applies_explicit_node_valid_mask_before_graph_postprocessing():
    h, out = make_inputs()
    model = DummyModel(costs=[0.1, 0.2, 0.9])

    nodes, edges = relation_infer(
        h,
        out,
        model,
        obj_token=3,
        rln_token=1,
        mode="mst",
        node_valid_mask=torch.tensor([[True, False, True]]),
    )

    np.testing.assert_allclose(nodes[0].numpy(), np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(edges[0], np.array([[0, 1]]))


def test_relation_infer_rejects_mismatched_node_valid_mask():
    h, out = make_inputs()

    with pytest.raises(ValueError, match="node_valid_mask"):
        relation_infer(
            h,
            out,
            DummyModel(costs=[0.1, 0.2, 0.9]),
            obj_token=3,
            rln_token=1,
            node_valid_mask=torch.tensor([[True, False]]),
        )


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


def test_virtual_root_mst_splits_two_components():
    node_pairs = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=torch.long,
    )
    edge_scores = torch.tensor([0.95, 0.05, 0.05, 0.05, 0.05, 0.95], dtype=torch.float32)
    root_scores = torch.tensor([0.6, 0.6, 0.6, 0.6], dtype=torch.float32)

    result = compute_virtual_root_forest_edges(node_pairs, edge_scores, root_scores)

    np.testing.assert_array_equal(result.real_edges, np.array([[0, 1], [2, 3]]))
    assert len(result.root_edges_node_indices) == 2
    np.testing.assert_array_equal(result.component_id, np.array([0, 0, 1, 1]))


def test_virtual_root_lambda_controls_split_merge():
    node_pairs = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=torch.long,
    )
    edge_scores = torch.tensor([0.95, 0.05, 0.05, 0.45, 0.05, 0.95], dtype=torch.float32)
    root_scores = torch.tensor([0.6, 0.6, 0.6, 0.6], dtype=torch.float32)

    split = compute_virtual_root_forest_edges(node_pairs, edge_scores, root_scores, root_penalty=0.0)
    merged = compute_virtual_root_forest_edges(node_pairs, edge_scores, root_scores, root_penalty=0.3)

    assert len(split.root_edges_node_indices) == 2
    assert len(merged.root_edges_node_indices) == 1
    np.testing.assert_array_equal(merged.real_edges, np.array([[0, 1], [1, 2], [2, 3]]))


def test_relation_infer_vr_mst_returns_forest_without_bridge():
    h = torch.zeros((1, 5, 2), dtype=torch.float32)
    out = {
        "pred_logits": torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        "pred_nodes": torch.tensor(
            [[[0.0, 0.0, 0.2, 0.2], [0.2, 0.0, 0.2, 0.2], [0.8, 0.0, 0.2, 0.2], [1.0, 0.0, 0.2, 0.2]]],
            dtype=torch.float32,
        ),
        "pred_root_logits": torch.ones((1, 4), dtype=torch.float32),
    }
    model = DummyModel(costs=[0.05, 0.95, 0.95, 0.95, 0.95, 0.05])

    nodes, edges, details = relation_infer(
        h,
        out,
        model,
        obj_token=4,
        rln_token=1,
        mode="vr-mst",
        return_details=True,
    )

    assert nodes[0].shape == (4, 2)
    np.testing.assert_array_equal(edges[0], np.array([[0, 1], [2, 3]]))
    np.testing.assert_array_equal(details[0]["component_id"], np.array([0, 0, 1, 1]))
    assert len(details[0]["root_edges_node_indices"]) == 2


def test_vr_mst_requires_root_logits():
    h, out = make_inputs()
    model = DummyModel(costs=[0.1, 0.2, 0.9])

    with pytest.raises(ValueError, match="pred_root_logits"):
        relation_infer(h, out, model, obj_token=3, rln_token=1, mode="vr-mst")
