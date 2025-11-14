"""
Inference module with Test-time constraint and optional distance weighting for TreeFormer.

This module implements the test-time constraint approach described in Section 5.3 of the paper,
with an optional distance weighting feature to incorporate geometric distances between nodes
into the MST computation.

Test-time Constraint:
    The test-time constraint ensures that the predicted graph forms a spanning tree structure
    by computing a Minimum Spanning Tree (MST) over the predicted edges. This guarantees that
    the output is a valid tree with N-1 edges for N nodes.

Distance Weighting:
    When enabled (use_distance=True), the cost function combines both the model's prediction
    of edge non-existence probability with the geometric distance between nodes:

    final_cost = (1 - α) * p(non-exist) + α * normalized_distance

    where:
        - p(non-exist): Probability that an edge does not exist (from model prediction)
        - normalized_distance: Euclidean distance normalized to [0, 1] range
        - α: distance_weight parameter controlling the balance between prediction and geometry

    This allows the MST to favor connections between nodes that are:
        1. Predicted to have edges by the model (low p(non-exist))
        2. Geometrically close to each other (low distance) when distance weighting is enabled

Functions:
    relation_infer: Main inference function with MST constraint and distance weighting
    compute_mst_nx: Helper function to compute MST using NetworkX library

References:
    Paper Section 5.3: "Test-time constraint"

Author: TreeFormer Project
"""

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import itertools
import networkx as nx
import numpy as np


def compute_mst_nx(node_pairs_valid, cost_pred_batch):
    """
    Compute Minimum Spanning Tree using NetworkX library.

    This function creates a weighted graph from the node pairs and their associated costs,
    then computes the MST using Kruskal's algorithm. The result is returned as an
    adjacency matrix.

    Args:
        node_pairs_valid (torch.Tensor): Tensor of shape (num_pairs, 2) containing valid
            node pair indices. Each row [i, j] represents a potential edge between
            nodes i and j.
        cost_pred_batch (torch.Tensor): Tensor of shape (num_pairs,) containing the
            cost/weight for each edge. Lower values indicate edges more likely to be
            included in the MST.

    Returns:
        torch.Tensor: Upper triangular adjacency matrix of shape (num_nodes, num_nodes)
            representing the MST. Non-zero entries indicate edges in the MST with their
            corresponding weights. The matrix is upper triangular (only entries where i < j
            are non-zero) to avoid redundancy.

    Notes:
        - Uses Kruskal's algorithm for MST computation via NetworkX
        - The cost values should represent the "cost" of including an edge (lower is better)
        - Returns an upper triangular matrix for consistency with the codebase
    """
    # Create NetworkX graph
    G = nx.Graph()

    # Convert tensors to numpy for efficient processing
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    # Add weighted edges to the graph
    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    # Compute MST using Kruskal's algorithm
    mst_edges = nx.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    mst_edges = list(mst_edges)

    # Create MST adjacency matrix
    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight

    # Convert to torch tensor and make upper triangular
    mst_adj_batch = torch.tensor(mst_adj_np)
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch


def relation_infer(h, out, net, obj_token, rln_token, nms=False, map_=False,
                   use_distance=False, distance_weight=0.5):
    """
    Infer relations between nodes with MST constraint and optional distance weighting.

    This function performs relation inference by:
    1. Identifying valid nodes from model predictions
    2. Optionally applying Non-Maximum Suppression (NMS) to valid nodes
    3. For each valid node pair, predicting relation existence probability
    4. Optionally incorporating geometric distances into the cost function
    5. Computing MST over the cost graph to ensure tree structure
    6. Returning predicted nodes and edges

    Args:
        h (torch.Tensor): Hidden states from transformer, shape (batch_size, num_tokens, hidden_dim).
            Contains embeddings for object tokens and relation token.
        out (dict): Model output dictionary containing:
            - 'pred_logits': Node validity predictions, shape (batch_size, num_tokens, 2)
            - 'pred_nodes': Node position predictions, shape (batch_size, num_tokens, 4)
                           Format: [x, y, width, height] where x, y are normalized coordinates
        net (torch.nn.Module): The neural network model, must have a 'relation_embed' module
            for predicting edge existence from concatenated node embeddings.
        obj_token (int): Number of object tokens in the sequence.
        rln_token (int): Number of relation tokens in the sequence (typically 1 or 0).
        nms (bool, optional): Whether to apply Non-Maximum Suppression on valid nodes.
            Defaults to False. When True, overlapping nodes are filtered based on IoU.
        map_ (bool, optional): Whether to return additional mapping information (boxes,
            scores, classes). Defaults to False.
        use_distance (bool, optional): Whether to incorporate geometric distances into the
            cost function. Defaults to False. When True, costs combine model predictions
            with normalized Euclidean distances.
        distance_weight (float, optional): Weight factor α for distance in cost computation.
            Defaults to 0.5. Valid range [0, 1] where:
            - 0: Only use model predictions (no distance influence)
            - 1: Only use geometric distances (ignore model predictions)
            - 0.5: Equal balance between predictions and distances

    Returns:
        If map_=False (default):
            tuple: (pred_nodes, pred_edges) where:
                - pred_nodes: List of torch.Tensors, one per batch. Each tensor has shape
                             (num_valid_nodes, 2) containing [x, y] coordinates.
                - pred_edges: List of numpy arrays, one per batch. Each array has shape
                             (num_edges, 2) containing edge indices in the MST.

        If map_=True:
            tuple: (pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score,
                   pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class)
                Additional returns for evaluation:
                - pred_nodes_boxes: Node bounding boxes (x, y, w, h)
                - pred_nodes_boxes_score: Node confidence scores
                - pred_nodes_boxes_class: Node class predictions
                - pred_edges_boxes_score: Edge confidence scores
                - pred_edges_boxes_class: Edge class predictions

    Notes:
        Distance Weighting Formula:
            When use_distance=True:
            1. Compute pairwise Euclidean distances: D = cdist(node_coords, node_coords)
            2. Normalize distances: D_norm = (D - D_min) / (D_max - D_min)
            3. Combine with predictions: cost = (1-α) * p(non-exist) + α * D_norm

        MST Computation:
            - The MST is computed to minimize total cost across all edges
            - Lower costs favor edge inclusion in the MST
            - Ensures exactly (N-1) edges for N nodes (tree property)

        Relation Feature Construction:
            - For each node pair (i, j), concatenates: [embed_i, embed_j, relation_token]
            - Computes bidirectional predictions and averages them for symmetry
    """
    # Extract object tokens from hidden states
    # Shape: (batch_size, obj_token, hidden_dim)
    object_token = h[..., :obj_token, :]

    # Extract relation token if present
    # Shape: (batch_size, rln_token, hidden_dim)
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    # Identify valid tokens (nodes that should be included)
    # valid_token: binary tensor indicating which tokens are valid
    # Shape: (batch_size, obj_token)
    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    # Apply Non-Maximum Suppression if requested
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            # Get indices of valid tokens
            valid_token_id = torch.nonzero(token).squeeze(1)

            # Extract logits and nodes for valid tokens
            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # Convert node format to bounding boxes for NMS
            # 0 <= x1 < x2 and 0 <= y1 < y2 must be fulfilled
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            # Apply batched NMS with high IoU threshold
            ids2keep = batched_nms(
                boxes=valid_nodes * 1000,
                scores=valid_scores,
                idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]

            # Update valid tokens after NMS
            valid_token_nms[idx][valid_token_id_nms] = 1
        valid_token = valid_token_nms

    # Initialize output lists
    pred_nodes = []
    pred_edges = []
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    # Process each batch independently
    for batch_id in range(h.shape[0]):
        # Get indices of valid nodes for this batch
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        # Store node coordinates (x, y only, ignore width/height)
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        # Store additional mapping information if requested
        if map_:
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        # Process edges only if we have at least 2 valid nodes
        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            # Generate all possible node pairs
            # node_pairs: list of [node_i_id, node_j_id] in original token space
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))

            # Generate node pairs in reindexed space (0 to num_valid_nodes-1)
            # Shape: (num_pairs, 2)
            node_pairs_valid = torch.tensor([
                list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))
            ])

            # Create mapping from node pair to pair index
            node_pairs_valid_dict = {}
            for num in range(node_pairs_valid.shape[0]):
                node_pair = node_pairs_valid[num]
                node_pairs_valid_dict[tuple(node_pair.cpu().numpy().tolist())] = num

            # Construct relation features for all node pairs
            if rln_token > 0:
                # Concatenate: [node_i_embed, node_j_embed, relation_token]
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
                # Concatenate: [node_j_embed, node_i_embed, relation_token]
                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                # Without relation token: [node_i_embed, node_j_embed]
                relation_feature1 = torch.cat(
                    (object_token[batch_id, node_pairs[0], :],
                     object_token[batch_id, node_pairs[1], :]), 1
                )
                relation_feature2 = torch.cat(
                    (object_token[batch_id, node_pairs[1], :],
                     object_token[batch_id, node_pairs[0], :]), 1
                )

            # Predict relation existence for both directions
            relation_pred1 = net.module.relation_embed(relation_feature1).detach()
            relation_pred2 = net.module.relation_embed(relation_feature2).detach()
            # Average predictions for symmetry
            relation_pred = (relation_pred1 + relation_pred2) / 2.0

            # Convert to probabilities
            # relation_pred_softmax_batch[:, 0] = p(non-exist)
            # relation_pred_softmax_batch[:, 1] = p(exist)
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1).detach()
            cost_pred_batch = relation_pred_softmax_batch[:, 0]  # Cost = p(non-exist)

            # Incorporate distance weighting if requested
            if use_distance:
                # Get node coordinates for distance computation
                # Shape: (num_valid_nodes, 2)
                node_coords = out['pred_nodes'][batch_id, node_id, :2].detach()

                # Compute pairwise Euclidean distances
                # Shape: (num_valid_nodes, num_valid_nodes)
                distance_matrix = torch.cdist(node_coords.unsqueeze(0),
                                             node_coords.unsqueeze(0),
                                             p=2).squeeze(0)

                # Extract distances for valid pairs only
                # Shape: (num_pairs,)
                x, y = node_pairs_valid.t()
                pairwise_distances = distance_matrix[x, y]

                # Normalize distances to [0, 1] range
                if pairwise_distances.max() > pairwise_distances.min():
                    normalized_distances = (pairwise_distances - pairwise_distances.min()) / \
                                         (pairwise_distances.max() - pairwise_distances.min())
                else:
                    # All distances are the same, set to 0
                    normalized_distances = torch.zeros_like(pairwise_distances)

                # Combine cost with distance
                # final_cost = (1 - α) * p(non-exist) + α * normalized_distance
                # Lower cost = more likely to be in MST
                cost_pred_batch = (1 - distance_weight) * cost_pred_batch + \
                                 distance_weight * normalized_distances

            # Compute MST using the cost function
            mst_adj_batch = compute_mst_nx(node_pairs_valid, cost_pred_batch)

            # Extract edges from MST adjacency matrix
            # Get indices of non-zero entries (edges in the MST)
            mst_tree_selected_list = torch.nonzero(mst_adj_batch, as_tuple=False)

            # Store edges as numpy array
            pred_edges.append(mst_tree_selected_list.cpu().numpy())

            # Map edge indices back to original node pair indices for scoring
            pred_rel_list = [
                node_pairs_valid_dict[tuple(sorted((int(xy[0]), int(xy[1]))))]
                for xy in mst_tree_selected_list if xy[0] != xy[1]
            ]
            pred_rel = torch.tensor(pred_rel_list).cpu().numpy()

            # Store additional edge information if requested
            if map_:
                pred_edges_boxes_score.append(
                    relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy()
                )
                pred_edges_boxes_class.append(
                    torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy()
                )
        else:
            # No edges if less than 2 valid nodes
            pred_edges.append(torch.empty(0, 2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0, 1).cpu().numpy())
                pred_edges_boxes_class.append(torch.empty(0, 1).cpu().numpy())

    # Return results based on map_ flag
    if map_:
        return (pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score,
                pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class)
    else:
        return pred_nodes, pred_edges
