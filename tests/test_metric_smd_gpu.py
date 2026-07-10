import numpy as np
import pytest
import torch

from metric_smd_gpu import sample_graph_point_cloud


def test_sample_graph_point_cloud_is_uniform_and_on_input_device() -> None:
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    cloud = sample_graph_point_cloud(nodes, torch.tensor([[0, 1], [1, 2]]), n_points=5)

    assert cloud.shape == (5, 2)
    assert cloud.device == nodes.device
    assert torch.allclose(cloud[:, 0], torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0]))
    assert torch.allclose(cloud[:, 1], torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0]))


def test_sample_graph_point_cloud_handles_numpy_empty_edges() -> None:
    nodes = torch.tensor([[0.2, 0.3]])
    cloud = sample_graph_point_cloud(nodes, np.empty((0, 2), dtype=np.int64), n_points=4)

    assert torch.equal(cloud, torch.zeros((4, 2)))


def test_geomloss_backend_is_optional_and_finite() -> None:
    geomloss = pytest.importorskip("geomloss")
    from metric_smd_gpu import GeomLossStreetMoverDistance

    del geomloss
    metric = GeomLossStreetMoverDistance(n_points=16, blur=0.05)
    nodes = [torch.tensor([[0.0, 0.0], [1.0, 0.0]])]
    edges = [torch.tensor([[0, 1]])]
    pred_nodes = [torch.tensor([[0.0, 0.0], [0.8, 0.0]])]
    pred_edges = [np.asarray([[0, 1]], dtype=np.int64)]
    result = metric(nodes, edges, pred_nodes, pred_edges)

    assert result.shape == (1,)
    assert torch.isfinite(result).all()
