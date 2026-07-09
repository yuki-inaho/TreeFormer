import torch
from monai.utils import MetricReduction

from metric_smd import StreetMoverDistance


def test_smd_accepts_forest_edges():
    metric = StreetMoverDistance(eps=1e-7, max_iter=10, reduction=MetricReduction.MEAN)
    nodes = [torch.tensor([[0.1, 0.1], [0.2, 0.1], [0.8, 0.1], [0.9, 0.1]], dtype=torch.float32)]
    edges = [torch.tensor([[0, 1], [2, 3]], dtype=torch.long)]
    pred_nodes = [nodes[0].clone()]
    pred_edges = [edges[0].clone()]

    value = metric(nodes, edges, pred_nodes, pred_edges)

    assert value.shape == (1,)
    assert torch.isfinite(value).all()
