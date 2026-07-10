"""Optional GPU implementation of the TreeFormer SMD validation metric.

The legacy metric is kept in :mod:`metric_smd` for compatibility.  This module
only replaces the point-cloud sampling and Sinkhorn distance; graph
post-processing remains the caller's responsibility.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_edge_tensor(edges: Any, *, device: torch.device) -> torch.Tensor:
    if edges is None:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    tensor = torch.as_tensor(edges, dtype=torch.long, device=device)
    return tensor.reshape(-1, 2)


def sample_graph_point_cloud(
    nodes: torch.Tensor,
    edges: Any,
    *,
    n_points: int = 500,
) -> torch.Tensor:
    """Sample a graph polyline uniformly by cumulative edge length on-device.

    The legacy implementation walks adjacency matrices in Python and emits a
    variable number of points.  This implementation always emits ``n_points``
    and uses tensor operations, which makes it suitable for batched GPU
    Sinkhorn evaluation.  It intentionally does not alter graph topology.
    """

    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    nodes = torch.as_tensor(nodes, dtype=torch.float32)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape [N, 2], got {tuple(nodes.shape)}")
    device = nodes.device
    edge_tensor = _as_edge_tensor(edges, device=device)
    if nodes.shape[0] == 0 or edge_tensor.shape[0] == 0:
        return torch.zeros((n_points, 2), dtype=nodes.dtype, device=device)
    if bool((edge_tensor < 0).any()) or bool((edge_tensor >= nodes.shape[0]).any()):
        raise ValueError("graph edge index is outside the node range")

    start = nodes[edge_tensor[:, 0]]
    delta = nodes[edge_tensor[:, 1]] - start
    lengths = torch.linalg.vector_norm(delta, dim=1)
    cumulative = torch.cumsum(lengths, dim=0)
    total_length = cumulative[-1]
    if bool(total_length <= torch.finfo(nodes.dtype).eps):
        return nodes[:1].expand(n_points, -1).clone()

    distances = torch.linspace(0.0, total_length, n_points, device=device, dtype=nodes.dtype)
    segment = torch.bucketize(distances, cumulative, right=True).clamp_max(edge_tensor.shape[0] - 1)
    previous = torch.cat((torch.zeros(1, device=device, dtype=nodes.dtype), cumulative[:-1]))
    local_distance = distances - previous[segment]
    fraction = (local_distance / lengths[segment].clamp_min(torch.finfo(nodes.dtype).eps)).clamp(0.0, 1.0)
    return start[segment] + fraction.unsqueeze(1) * delta[segment]


class GeomLossStreetMoverDistance:
    """Opt-in GPU SMD approximation backed by ``geomloss.SamplesLoss``.

    ``debias=False`` keeps the objective closer to the legacy regularized
    transport cost.  The result is not promised to be bitwise equal to the
    legacy CPU Sinkhorn implementation; callers should compare ranking and
    validation behavior before using it for checkpoint selection.
    """

    def __init__(
        self,
        *,
        n_points: int = 500,
        blur: float = 0.01,
        backend: str = "tensorized",
    ) -> None:
        try:
            from geomloss import SamplesLoss
        except ImportError as exc:
            raise ImportError(
                "geomloss is required for SMD_BACKEND=geomloss_gpu; "
                "install it with `uv pip install --python .venv/bin/python 'geomloss>=0.2.6,<0.3'`."
            ) from exc
        if n_points <= 0:
            raise ValueError(f"n_points must be positive, got {n_points}")
        if blur <= 0.0:
            raise ValueError(f"blur must be positive, got {blur}")
        self.n_points = int(n_points)
        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=float(blur),
            debias=False,
            backend=str(backend),
        )
        self._values: list[torch.Tensor] = []

    def __call__(self, node_list, edge_list, pred_node_list, pred_edge_list):  # type: ignore[no-untyped-def]
        values: list[torch.Tensor] = []
        for nodes, edges, pred_nodes, pred_edges in zip(node_list, edge_list, pred_node_list, pred_edge_list):
            nodes = torch.as_tensor(nodes, dtype=torch.float32)
            pred_nodes = torch.as_tensor(pred_nodes, dtype=torch.float32, device=nodes.device)
            edge_tensor = _as_edge_tensor(edges, device=nodes.device)
            pred_edge_tensor = _as_edge_tensor(pred_edges, device=pred_nodes.device)
            if nodes.shape[0] <= 1 or pred_nodes.shape[0] <= 1 or pred_edge_tensor.shape[0] == 0:
                values.append(torch.ones((), dtype=torch.float32, device=pred_nodes.device))
                continue
            target_cloud = sample_graph_point_cloud(nodes, edge_tensor, n_points=self.n_points)
            pred_cloud = sample_graph_point_cloud(pred_nodes, pred_edge_tensor, n_points=self.n_points)
            values.append(self.sinkhorn(target_cloud.unsqueeze(0), pred_cloud.unsqueeze(0)).reshape(()))
        if not values:
            return torch.empty((0,), dtype=torch.float32)
        result = torch.stack(values)
        self._values.append(result.detach())
        return result

    def reset(self) -> None:
        self._values.clear()
