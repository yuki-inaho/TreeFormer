import itertools
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms


InferenceReturn = Union[
    Tuple[List[torch.Tensor], List[np.ndarray]],
    Tuple[List[torch.Tensor], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]


def get_relation_embed(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = getattr(model, "module", model)
    return wrapped.relation_embed


def compute_mst_edges(node_pairs_valid: torch.Tensor, cost_pred_batch: torch.Tensor) -> np.ndarray:
    graph = nx.Graph()
    pairs = node_pairs_valid.detach().cpu().numpy()
    costs = cost_pred_batch.detach().cpu().numpy()
    graph.add_weighted_edges_from((int(u), int(v), float(w)) for (u, v), w in zip(pairs, costs))

    mst_edges = []
    for u, v in nx.minimum_spanning_edges(graph, algorithm="kruskal", data=False):
        mst_edges.append((min(int(u), int(v)), max(int(u), int(v))))

    if not mst_edges:
        return np.empty((0, 2), dtype=np.int64)

    return np.array(sorted(mst_edges), dtype=np.int64)


def _valid_tokens_after_nms(valid_token: torch.Tensor, out: dict) -> torch.Tensor:
    valid_token_nms = torch.zeros_like(valid_token)
    for idx, (token, logits, nodes) in enumerate(zip(valid_token, out["pred_logits"], out["pred_nodes"])):
        valid_token_id = torch.nonzero(token).squeeze(1)
        if valid_token_id.numel() == 0:
            continue

        valid_logits = logits[valid_token_id]
        valid_nodes = nodes[valid_token_id].clone()
        valid_scores = F.softmax(valid_logits, dim=1)[:, 1]
        valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5
        ids_to_keep = batched_nms(
            boxes=valid_nodes * 1000,
            scores=valid_scores,
            idxs=torch.ones_like(valid_scores, dtype=torch.long),
            iou_threshold=0.90,
        )
        valid_token_nms[idx][valid_token_id[ids_to_keep].sort()[0]] = 1

    return valid_token_nms


def _pair_relation_logits(
    h: torch.Tensor,
    model: torch.nn.Module,
    batch_id: int,
    node_id: torch.Tensor,
    obj_token: int,
    rln_token: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    object_token = h[..., :obj_token, :]
    local_pairs = torch.tensor(
        list(itertools.combinations(range(node_id.shape[0]), 2)),
        dtype=torch.long,
        device=h.device,
    )
    token_pairs = node_id[local_pairs]

    if rln_token > 0:
        relation_token = h[..., obj_token : obj_token + rln_token, :]
        relation_context = relation_token[batch_id].reshape(1, -1).repeat(local_pairs.shape[0], 1)
        relation_feature1 = torch.cat(
            (
                object_token[batch_id, token_pairs[:, 0], :],
                object_token[batch_id, token_pairs[:, 1], :],
                relation_context,
            ),
            dim=1,
        )
        relation_feature2 = torch.cat(
            (
                object_token[batch_id, token_pairs[:, 1], :],
                object_token[batch_id, token_pairs[:, 0], :],
                relation_context,
            ),
            dim=1,
        )
    else:
        relation_feature1 = torch.cat(
            (object_token[batch_id, token_pairs[:, 0], :], object_token[batch_id, token_pairs[:, 1], :]),
            dim=1,
        )
        relation_feature2 = torch.cat(
            (object_token[batch_id, token_pairs[:, 1], :], object_token[batch_id, token_pairs[:, 0], :]),
            dim=1,
        )

    relation_embed = get_relation_embed(model)
    relation_pred1 = relation_embed(relation_feature1).detach()
    relation_pred2 = relation_embed(relation_feature2).detach()
    return local_pairs, (relation_pred1 + relation_pred2) / 2.0


def _distance_weighted_cost(
    out: dict,
    batch_id: int,
    node_id: torch.Tensor,
    local_pairs: torch.Tensor,
    cost_pred_batch: torch.Tensor,
    distance_weight: float,
) -> torch.Tensor:
    if not 0 <= distance_weight <= 1:
        raise ValueError("distance_weight must be in [0, 1]")

    node_coords = out["pred_nodes"][batch_id, node_id, :2].detach()
    distance_matrix = torch.cdist(node_coords.unsqueeze(0), node_coords.unsqueeze(0), p=2).squeeze(0)
    pairwise_distances = distance_matrix[local_pairs[:, 0], local_pairs[:, 1]]

    if pairwise_distances.max() > pairwise_distances.min():
        distances = (pairwise_distances - pairwise_distances.min()) / (
            pairwise_distances.max() - pairwise_distances.min()
        )
    else:
        distances = torch.zeros_like(pairwise_distances)

    return (1 - distance_weight) * cost_pred_batch + distance_weight * distances


def relation_infer(
    h: torch.Tensor,
    out: dict,
    model: torch.nn.Module,
    obj_token: int,
    rln_token: int,
    nms: bool = False,
    map_: bool = False,
    mst: bool = False,
    use_distance: bool = False,
    distance_weight: float = 0.5,
) -> InferenceReturn:
    valid_token = torch.argmax(out["pred_logits"], -1).detach()
    if nms:
        valid_token = _valid_tokens_after_nms(valid_token, out)

    pred_nodes = []
    pred_edges = []
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        pred_nodes.append(out["pred_nodes"][batch_id, node_id, :2].detach())

        if map_:
            pred_nodes_boxes.append(out["pred_nodes"][batch_id, node_id, :].detach().cpu().numpy())
            pred_nodes_boxes_score.append(out["pred_logits"].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy())
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        if node_id.numel() <= 1:
            pred_edges.append(np.empty((0, 2), dtype=np.int64))
            if map_:
                pred_edges_boxes_score.append(np.empty((0,), dtype=np.float32))
                pred_edges_boxes_class.append(np.empty((0,), dtype=np.int64))
            continue

        local_pairs, relation_pred = _pair_relation_logits(h, model, batch_id, node_id, obj_token, rln_token)

        if mst:
            cost_pred_batch = F.softmax(relation_pred, dim=-1)[:, 0]
            if use_distance:
                cost_pred_batch = _distance_weighted_cost(
                    out, batch_id, node_id, local_pairs, cost_pred_batch, distance_weight
                )
            selected_edges = compute_mst_edges(local_pairs, cost_pred_batch)
            selected_rel = []
            pair_to_index = {
                tuple(pair.tolist()): idx for idx, pair in enumerate(local_pairs.detach().cpu())
            }
            for edge in selected_edges:
                selected_rel.append(pair_to_index[tuple(edge.tolist())])
            pred_rel = torch.tensor(selected_rel, dtype=torch.long, device=relation_pred.device)
        else:
            pred_rel = torch.nonzero(torch.argmax(relation_pred, -1)).squeeze(1)
            selected_edges = local_pairs[pred_rel].detach().cpu().numpy()

        pred_edges.append(selected_edges)

        if map_:
            pred_edges_boxes_score.append(relation_pred.softmax(-1)[pred_rel, 1].detach().cpu().numpy())
            pred_edges_boxes_class.append(torch.argmax(relation_pred, -1)[pred_rel].detach().cpu().numpy())

    if map_:
        return (
            pred_nodes,
            pred_edges,
            pred_nodes_boxes,
            pred_nodes_boxes_score,
            pred_nodes_boxes_class,
            pred_edges_boxes_score,
            pred_edges_boxes_class,
        )

    return pred_nodes, pred_edges
