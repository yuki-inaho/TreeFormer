from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


LEGACY_SINGLE_TREE = "legacy_single_tree"
VIRTUAL_ROOT_FOREST_V1 = "virtual_root_forest_v1"


@dataclass(frozen=True)
class VirtualRootForestResult:
    real_edges: np.ndarray
    root_edges_node_indices: np.ndarray
    component_id: np.ndarray
    augmented_edges: np.ndarray


def _empty_edges() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def _as_numpy_edges(edges: torch.Tensor | np.ndarray | Any) -> np.ndarray:
    if isinstance(edges, torch.Tensor):
        edges = edges.detach().cpu().numpy()
    array = np.asarray(edges, dtype=np.int64)
    if array.size == 0:
        return _empty_edges()
    return array.reshape(-1, 2)


def _as_numpy_vector(values: torch.Tensor | np.ndarray | Any, *, dtype: np.dtype = np.int64) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype).reshape(-1)


def _read_field(datapoint: Any, name: str, default: Any = None) -> Any:
    if isinstance(datapoint, Mapping):
        return datapoint.get(name, default)
    return getattr(datapoint, name, default)


def infer_component_id(num_nodes: int, edges: torch.Tensor | np.ndarray | Any) -> torch.Tensor:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    for start, end in _as_numpy_edges(edges).tolist():
        if 0 <= start < num_nodes and 0 <= end < num_nodes:
            graph.add_edge(int(start), int(end))

    component_id = torch.empty((int(num_nodes),), dtype=torch.long)
    for component_index, nodes in enumerate(sorted(nx.connected_components(graph), key=lambda item: min(item) if item else -1)):
        for node in nodes:
            component_id[int(node)] = int(component_index)
    return component_id


def root_nodes_from_component_id(component_id: torch.Tensor) -> torch.Tensor:
    component_id = component_id.to(dtype=torch.long).reshape(-1)
    if component_id.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    roots = []
    for component in torch.unique(component_id, sorted=True).tolist():
        candidates = torch.nonzero(component_id == int(component)).reshape(-1)
        roots.append(candidates.min())
    return torch.stack(roots).to(dtype=torch.long)


def root_edge_index_from_roots(root_node_indices: torch.Tensor) -> torch.Tensor:
    roots = root_node_indices.to(dtype=torch.long).reshape(-1)
    if roots.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.stack((torch.zeros_like(roots), roots + 1), dim=1)


def load_forest_metadata(
    datapoint: Any,
    *,
    nodes: torch.Tensor,
    edges: torch.Tensor,
    strict_virtual_root: bool = False,
) -> dict[str, Any]:
    """Read optional virtual-root forest metadata while preserving legacy samples."""

    node_count = int(nodes.shape[0])
    graph_topology = _read_field(datapoint, "graph_topology", None)
    component_id_raw = _read_field(datapoint, "component_id", None)
    root_node_indices_raw = _read_field(datapoint, "root_node_indices", None)
    component_count_raw = _read_field(datapoint, "component_count", None)
    root_edge_index_raw = _read_field(datapoint, "root_edge_index", None)

    explicit_virtual_root = graph_topology == VIRTUAL_ROOT_FOREST_V1
    if strict_virtual_root or explicit_virtual_root:
        missing = []
        if component_id_raw is None:
            missing.append("component_id")
        if missing:
            raise ValueError(f"virtual-root forest metadata requires fields: {', '.join(missing)}")

    if component_id_raw is None:
        component_id = infer_component_id(node_count, edges)
        graph_topology = graph_topology or LEGACY_SINGLE_TREE
    else:
        component_id = torch.as_tensor(component_id_raw, dtype=torch.long).reshape(-1)
        if component_id.numel() != node_count:
            raise ValueError(f"component_id length must match node count: {component_id.numel()} != {node_count}")
        graph_topology = graph_topology or VIRTUAL_ROOT_FOREST_V1

    if component_count_raw is None:
        component_count = int(torch.unique(component_id).numel()) if component_id.numel() else 0
    else:
        component_count = int(torch.as_tensor(component_count_raw).item())

    root_node_indices = (
        root_nodes_from_component_id(component_id)
        if root_node_indices_raw is None
        else torch.as_tensor(root_node_indices_raw, dtype=torch.long).reshape(-1)
    )
    root_edge_index = (
        root_edge_index_from_roots(root_node_indices)
        if root_edge_index_raw is None
        else torch.as_tensor(root_edge_index_raw, dtype=torch.long).reshape(-1, 2)
    )

    return {
        "component_id": component_id,
        "component_count": component_count,
        "root_node_indices": root_node_indices,
        "root_edge_index": root_edge_index,
        "graph_topology": str(graph_topology),
    }


def compute_virtual_root_forest_edges(
    real_pairs: torch.Tensor | np.ndarray | Any,
    real_edge_scores: torch.Tensor | np.ndarray | Any,
    root_scores: torch.Tensor | np.ndarray | Any,
    *,
    root_penalty: float = 0.0,
) -> VirtualRootForestResult:
    """Run maximum spanning tree on V union {virtual root} and return the real-node forest."""

    pairs = _as_numpy_edges(real_pairs)
    edge_scores = _as_numpy_vector(real_edge_scores, dtype=np.float64)
    root_scores_np = _as_numpy_vector(root_scores, dtype=np.float64)
    node_count = int(root_scores_np.shape[0])
    if pairs.shape[0] != edge_scores.shape[0]:
        raise ValueError(f"real_pairs and real_edge_scores length mismatch: {pairs.shape[0]} != {edge_scores.shape[0]}")

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count + 1))
    for index, ((start, end), score) in enumerate(zip(pairs, edge_scores)):
        if not (0 <= start < node_count and 0 <= end < node_count):
            raise ValueError(f"real edge index out of range for {node_count} nodes: {(int(start), int(end))}")
        graph.add_edge(int(start) + 1, int(end) + 1, weight=float(score), order=int(index))
    for index, score in enumerate(root_scores_np):
        graph.add_edge(0, int(index) + 1, weight=float(score) - float(root_penalty), order=int(pairs.shape[0] + index))

    if node_count == 0:
        return VirtualRootForestResult(
            real_edges=_empty_edges(),
            root_edges_node_indices=np.empty((0,), dtype=np.int64),
            component_id=np.empty((0,), dtype=np.int64),
            augmented_edges=_empty_edges(),
        )

    tree_edges = list(nx.maximum_spanning_edges(graph, algorithm="kruskal", data=False))
    real_edges: list[tuple[int, int]] = []
    root_edges: list[int] = []
    augmented_edges: list[tuple[int, int]] = []
    forest = nx.Graph()
    forest.add_nodes_from(range(node_count))
    for start, end in tree_edges:
        u, v = int(start), int(end)
        augmented_edges.append((min(u, v), max(u, v)))
        if u == 0 or v == 0:
            root_edges.append((v if u == 0 else u) - 1)
            continue
        real_start, real_end = u - 1, v - 1
        edge = (min(real_start, real_end), max(real_start, real_end))
        real_edges.append(edge)
        forest.add_edge(*edge)

    component_id = np.empty((node_count,), dtype=np.int64)
    for component_index, nodes in enumerate(sorted(nx.connected_components(forest), key=lambda item: min(item))):
        for node in nodes:
            component_id[int(node)] = int(component_index)

    return VirtualRootForestResult(
        real_edges=np.asarray(sorted(real_edges), dtype=np.int64).reshape(-1, 2) if real_edges else _empty_edges(),
        root_edges_node_indices=np.asarray(sorted(root_edges), dtype=np.int64),
        component_id=component_id,
        augmented_edges=np.asarray(sorted(augmented_edges), dtype=np.int64).reshape(-1, 2) if augmented_edges else _empty_edges(),
    )


def build_cross_component_edge_labels(
    node_count: int,
    positive_edges: torch.Tensor,
    component_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if component_id is None:
        raise ValueError("virtual-root edge labels require component_id")
    component_id = component_id.to(dtype=torch.long).reshape(-1)
    if int(component_id.numel()) != int(node_count):
        raise ValueError(f"component_id length must match node count: {component_id.numel()} != {node_count}")

    positive = {
        tuple(sorted((int(start), int(end))))
        for start, end in positive_edges.to(dtype=torch.long).reshape(-1, 2).detach().cpu().tolist()
    }
    pairs = []
    labels = []
    for start in range(int(node_count)):
        for end in range(start + 1, int(node_count)):
            pairs.append((start, end))
            if component_id[start].item() != component_id[end].item():
                labels.append(0)
            else:
                labels.append(1 if (start, end) in positive else 0)
    device = component_id.device
    return (
        torch.tensor(pairs, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def root_mil_loss(
    root_logits: torch.Tensor,
    component_id: torch.Tensor,
    *,
    cardinality_weight: float = 0.05,
) -> torch.Tensor:
    if component_id is None:
        raise ValueError("virtual-root root loss requires component_id")
    root_logits = root_logits.reshape(-1)
    component_id = component_id.to(device=root_logits.device, dtype=torch.long).reshape(-1)
    if root_logits.numel() != component_id.numel():
        raise ValueError(f"root_logits length must match component_id length: {root_logits.numel()} != {component_id.numel()}")
    if root_logits.numel() == 0:
        return root_logits.sum()

    probabilities = torch.sigmoid(root_logits)
    losses = []
    for component in torch.unique(component_id, sorted=True):
        component_prob = probabilities[component_id == component]
        no_root_probability = torch.prod(1.0 - component_prob.clamp(min=1e-6, max=1.0 - 1e-6))
        losses.append(-torch.log1p(-no_root_probability.clamp(max=1.0 - 1e-6)))
    mil = torch.stack(losses).mean()
    expected_count = probabilities.sum()
    target_count = torch.as_tensor(float(len(losses)), dtype=root_logits.dtype, device=root_logits.device)
    cardinality = F.smooth_l1_loss(expected_count, target_count)
    return mil + float(cardinality_weight) * cardinality
