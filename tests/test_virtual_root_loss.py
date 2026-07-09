import pytest
import torch

from treeformer_train.virtual_root import build_cross_component_edge_labels, root_mil_loss


def test_root_mil_loss_uses_one_root_per_component():
    component_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    good_logits = torch.tensor([4.0, -4.0, 4.0, -4.0], dtype=torch.float32)
    bad_logits = torch.tensor([-4.0, -4.0, -4.0, -4.0], dtype=torch.float32)

    good_loss = root_mil_loss(good_logits, component_id)
    bad_loss = root_mil_loss(bad_logits, component_id)

    assert good_loss.item() < bad_loss.item()


def test_cross_component_pairs_are_negative_edges():
    positive_edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    component_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    pairs, labels = build_cross_component_edge_labels(4, positive_edges, component_id)
    label_by_pair = {tuple(pair.tolist()): int(label.item()) for pair, label in zip(pairs, labels)}

    assert label_by_pair[(0, 1)] == 1
    assert label_by_pair[(2, 3)] == 1
    assert label_by_pair[(0, 2)] == 0
    assert label_by_pair[(1, 3)] == 0


def test_virtual_root_constraint_does_not_force_bridge():
    positive_edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    component_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    pairs, labels = build_cross_component_edge_labels(4, positive_edges, component_id)
    positive_pairs = {tuple(pair.tolist()) for pair, label in zip(pairs, labels) if int(label.item()) == 1}

    assert positive_pairs == {(0, 1), (2, 3)}
    assert (1, 2) not in positive_pairs


def test_virtual_root_loss_requires_component_id():
    with pytest.raises(ValueError, match="component_id"):
        build_cross_component_edge_labels(2, torch.empty((0, 2), dtype=torch.long), None)
