from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphSample:
    """RGB image and normalized graph annotation passed through augmentation."""

    image: np.ndarray
    nodes: np.ndarray
    edges: np.ndarray

    def copy(self) -> "GraphSample":
        return GraphSample(image=self.image.copy(), nodes=self.nodes.copy(), edges=self.edges.copy())


def as_graph_sample(image: np.ndarray, nodes: np.ndarray, edges: np.ndarray) -> GraphSample:
    return GraphSample(image=as_float_image(image), nodes=as_nodes(nodes), edges=as_edges(edges))


def as_float_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.size and image_array.max() > 1.5:
        image_array = image_array / 255.0
    return np.clip(image_array, 0.0, 1.0)


def as_nodes(nodes: np.ndarray) -> np.ndarray:
    node_array = np.asarray(nodes, dtype=np.float32)
    if node_array.size == 0:
        return node_array.reshape(0, 2)
    return node_array.reshape(-1, 2)


def as_edges(edges: np.ndarray) -> np.ndarray:
    edge_array = np.asarray(edges, dtype=np.int64)
    if edge_array.size == 0:
        return edge_array.reshape(0, 2)
    return edge_array.reshape(-1, 2)
