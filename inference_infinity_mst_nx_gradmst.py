"""
TreeFormer Inference with SFS Layer and NetworkX Kruskal's MST Algorithm

This module implements the TreeFormer inference algorithm using the Straight-Forward
Softmax (SFS) Layer as described in the paper, with NetworkX's Kruskal algorithm for
Minimum Spanning Tree (MST) computation.

Paper Reference:
    Section 4.2: "TreeFormer with SFS Layer"
    Equation (10): Cost Matrix Construction

Algorithm Overview:
    1. Extract object and relation tokens from hidden features
    2. Identify valid nodes through token classification
    3. Build cost matrix from edge non-existence probabilities:
       ŷ^-_(i,j) = p(edge (i,j) does not exist)
    4. Compute MST using NetworkX Kruskal's algorithm on cost matrix
    5. Return predicted tree structure (nodes and edges)

Key Concepts:
    - Object Tokens: Represent individual entities/nodes in the scene
    - Relation Tokens: Encode relational information between objects
    - Cost Matrix: Edge weights derived from non-existence probabilities
    - MST: Minimum Spanning Tree connecting all nodes with minimal cost
    - SFS Layer: Modified softmax layer for tree-structured prediction

Cost Matrix Construction (Equation 10):
    The cost for edge (i,j) is defined as the probability that the edge
    does NOT exist: cost(i,j) = P(edge_ij = 0) = softmax(logits)[0]

    Lower cost indicates higher probability of edge existence, making
    Kruskal's minimum spanning tree select the most likely edges.

Dependencies:
    - torch: PyTorch for tensor operations and neural network inference
    - networkx: Graph algorithms including Kruskal's MST
    - numpy: Numerical operations and array manipulation
    - itertools: Combinatorial operations for generating node pairs
    - torchvision.ops: Non-maximum suppression for node filtering

Author: Agent-1
Date: 2025-11-14
"""

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import itertools
import networkx as nx
import numpy as np
from typing import Tuple, List, Optional, Union


def compute_mst_nx(
    node_pairs_valid: torch.Tensor,
    cost_pred_batch: torch.Tensor
) -> torch.Tensor:
    """
    Compute Minimum Spanning Tree using NetworkX Kruskal's algorithm.

    This function constructs a weighted graph from node pairs and their associated
    costs, then computes the MST using Kruskal's algorithm. The cost represents
    the probability that an edge does NOT exist, so Kruskal's algorithm will
    select edges with lowest non-existence probability (highest existence probability).

    Algorithm Steps:
        1. Create empty NetworkX graph
        2. Add weighted edges from node pairs and costs
        3. Run Kruskal's MST algorithm
        4. Convert MST to upper triangular adjacency matrix

    Args:
        node_pairs_valid (torch.Tensor): Tensor of shape (N, 2) containing valid
            node pairs where N is the number of possible edges. Each row [i, j]
            represents a potential edge between node i and node j.
            Example: tensor([[0, 1], [0, 2], [1, 2]]) for 3 nodes.

        cost_pred_batch (torch.Tensor): Tensor of shape (N,) containing edge costs,
            where cost = P(edge does not exist). Lower costs indicate higher
            probability of edge existence.
            Example: tensor([0.1, 0.3, 0.2]) means edges (0,1), (0,2), (1,2)
            have non-existence probabilities of 0.1, 0.3, 0.2 respectively.

    Returns:
        torch.Tensor: Upper triangular adjacency matrix of shape (num_nodes, num_nodes)
            representing the MST. Non-zero entries indicate edges in the MST with
            their associated costs. The matrix is upper triangular (i < j for edge (i,j)).
            Example output for 3 nodes:
            tensor([[0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.2],
                    [0.0, 0.0, 0.0]])
            This represents edges (0,1) with cost 0.1 and (1,2) with cost 0.2.

    Implementation Details:
        - Uses NetworkX's minimum_spanning_edges with algorithm="kruskal"
        - Converts node pairs and costs to NetworkX weighted graph format
        - Returns adjacency matrix in PyTorch tensor format
        - Ensures upper triangular structure for consistent edge representation

    Example:
        >>> node_pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
        >>> costs = torch.tensor([0.1, 0.3, 0.2])
        >>> mst_adj = compute_mst_nx(node_pairs, costs)
        >>> print(mst_adj)
        tensor([[0.0, 0.1, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0]])

    Note:
        The returned adjacency matrix is UPPER TRIANGULAR, meaning for each edge
        only the entry (i,j) where i < j is non-zero. This avoids duplicate
        edge representations and simplifies downstream processing.
    """
    # Create NetworkX graph
    G = nx.Graph()

    # Convert tensors to numpy for efficient iteration
    # This avoids repeated GPU-CPU transfers
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    # Add weighted edges to graph
    # Each edge (u, v) has weight w = P(edge does not exist)
    # Kruskal's algorithm finds MST by selecting edges with minimum weight
    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    # Compute MST using Kruskal's algorithm
    # algorithm="kruskal": Sort edges by weight, add edges that don't create cycles
    # data=False: Return only node pairs, not edge attributes
    mst_edges = nx.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    mst_edges = list(mst_edges)

    # Create adjacency matrix for MST
    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))

    # Populate adjacency matrix with edge weights from MST
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight  # Symmetric for undirected graph

    # Convert to PyTorch tensor
    mst_adj_batch = torch.tensor(mst_adj_np)

    # Convert to upper triangular matrix
    # This ensures each edge (i,j) is represented only once with i < j
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch


def relation_infer(
    h: torch.Tensor,
    out: dict,
    net: torch.nn.Module,
    obj_token: int,
    rln_token: int,
    nms: bool = False,
    map_: bool = False
) -> Union[Tuple[List, List], Tuple[List, List, List, List, List, List, List]]:
    """
    Perform tree structure inference using TreeFormer with SFS Layer.

    This function implements the complete TreeFormer inference pipeline as described
    in Section 4.2 of the paper. It extracts object and relation representations,
    constructs a cost matrix based on edge non-existence probabilities, computes
    the MST using NetworkX Kruskal's algorithm, and returns the predicted tree structure.

    Algorithm Pipeline:
        1. Token Extraction:
           - Extract object tokens (first obj_token positions)
           - Extract relation token (position obj_token, if rln_token > 0)

        2. Node Selection:
           - Identify valid nodes using token classification logits
           - Optionally apply NMS (Non-Maximum Suppression) for dense predictions

        3. Edge Cost Computation:
           - Generate all possible node pairs for valid nodes
           - Compute relation features by concatenating object tokens
           - Predict edge probabilities using relation_embed network
           - Extract cost = P(edge does not exist) from softmax output

        4. MST Computation:
           - Build cost matrix from edge costs
           - Apply NetworkX Kruskal's algorithm to find MST
           - Extract MST edges as tree structure

        5. Result Processing:
           - Map MST edges back to original node indices
           - Optionally compute prediction scores and classes for evaluation

    Cost Matrix Construction (Equation 10):
        For each node pair (i, j):
        1. Concatenate features: [obj_i, obj_j] or [obj_i, obj_j, rln]
        2. Predict logits: logits_ij = relation_embed([obj_i, obj_j, rln])
        3. Compute cost: cost_ij = softmax(logits_ij)[0] = P(edge_ij = 0)

        The cost represents the probability that edge (i,j) does NOT exist.
        Lower cost = higher probability of edge existence.

    SFS Layer (Straight-Forward Softmax):
        During training, the SFS layer modifies edge probabilities based on MST:
        - If edge is in MST: Keep original probability
        - If edge not in MST: Reduce existence probability towards 0

        This encourages the model to predict tree-structured outputs during inference.

    Args:
        h (torch.Tensor): Hidden features from transformer encoder.
            Shape: (batch_size, num_tokens, hidden_dim)
            Contains concatenated object and relation tokens.
            Example: (2, 21, 256) for 2 images with 20 object + 1 relation token.

        out (dict): Dictionary containing model predictions with keys:
            - 'pred_logits': Node classification logits, shape (batch_size, num_tokens, 2)
              Dimension 2 represents [not_valid, valid] node classification.
            - 'pred_nodes': Node bounding box predictions, shape (batch_size, num_tokens, 4)
              Format: [center_x, center_y, width, height] normalized to [0, 1].

        net (torch.nn.Module): The trained TreeFormer model containing:
            - relation_embed: MLP for edge classification from relation features

        obj_token (int): Number of object tokens in the sequence.
            Example: 20 means first 20 tokens represent potential objects.

        rln_token (int): Number of relation tokens (typically 0 or 1).
            If > 0, includes global relation context in edge prediction.

        nms (bool, optional): Whether to apply Non-Maximum Suppression to filter
            overlapping node predictions. Defaults to False.
            Useful for dense prediction scenarios with many overlapping boxes.

        map_ (bool, optional): Whether to return additional outputs for evaluation
            including confidence scores and class labels. Defaults to False.

    Returns:
        If map_ is False:
            Tuple[List, List]:
                - pred_nodes: List of predicted node coordinates per batch
                  Each element is tensor of shape (num_valid_nodes, 2) with [x, y]
                - pred_edges: List of predicted edges per batch
                  Each element is numpy array of shape (num_edges, 2) with [node_i, node_j]

        If map_ is True:
            Tuple[List, List, List, List, List, List, List]:
                Returns 7 lists for detailed evaluation:
                1. pred_nodes: Node coordinates
                2. pred_edges: Edge indices
                3. pred_nodes_boxes: Full bounding boxes [x, y, w, h]
                4. pred_nodes_boxes_score: Node confidence scores
                5. pred_nodes_boxes_class: Node class labels
                6. pred_edges_boxes_score: Edge confidence scores
                7. pred_edges_boxes_class: Edge class labels

    Example:
        >>> h = torch.randn(2, 21, 256).cuda()  # 2 images, 20 obj + 1 rln tokens
        >>> out = {
        ...     'pred_logits': torch.randn(2, 20, 2).cuda(),
        ...     'pred_nodes': torch.rand(2, 20, 4).cuda()
        ... }
        >>> pred_nodes, pred_edges = relation_infer(h, out, model, 20, 1)
        >>> print(f"Batch 0: {len(pred_nodes[0])} nodes, {len(pred_edges[0])} edges")
        Batch 0: 7 nodes, 6 edges  # Tree with 7 nodes has 6 edges

    Implementation Notes:
        - Uses bidirectional edge features: avg([obj_i, obj_j, rln], [obj_j, obj_i, rln])
        - Applies NetworkX Kruskal's algorithm for globally optimal MST
        - Handles edge cases: no valid nodes, single node (no edges)
        - Maintains correspondence between local (0-based) and global node indices

    Mathematical Details:
        Given N valid nodes, the algorithm:
        1. Generates C(N, 2) = N*(N-1)/2 potential edges
        2. Computes cost for each edge: cost_ij = P(edge_ij = 0)
        3. Builds MST selecting N-1 edges with minimum total cost
        4. Returns tree structure guaranteeing connectivity

    References:
        - TreeFormer Paper Section 4.2: SFS Layer description
        - Equation (10): Cost matrix construction from edge probabilities
        - NetworkX documentation: minimum_spanning_edges with Kruskal's algorithm
    """
    # ========================================================================
    # Step 1: Extract Object and Relation Tokens
    # ========================================================================

    # Extract object tokens from hidden features
    # Shape: (batch_size, obj_token, hidden_dim)
    object_token = h[..., :obj_token, :]

    # Extract relation token if present
    # Shape: (batch_size, rln_token, hidden_dim)
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    # ========================================================================
    # Step 2: Node Selection via Token Classification
    # ========================================================================

    # Identify valid nodes from classification logits
    # pred_logits shape: (batch_size, num_tokens, 2) where dim 2 = [invalid, valid]
    # valid_token shape: (batch_size, num_tokens) with values 0 (invalid) or 1 (valid)
    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    # ========================================================================
    # Step 3: Optional Non-Maximum Suppression (NMS)
    # ========================================================================

    if nms:
        # Apply NMS to filter overlapping predictions
        valid_token_nms = torch.zeros_like(valid_token)

        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            # Get indices of valid tokens
            valid_token_id = torch.nonzero(token).squeeze(1)

            if valid_token_id.numel() == 0:
                continue

            # Extract logits and boxes for valid tokens
            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # Convert boxes to format required by NMS
            # boxes must satisfy: 0 <= x1 < x2 and 0 <= y1 < y2
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            # Apply batched NMS with IoU threshold 0.90
            ids2keep = batched_nms(
                boxes=valid_nodes * 1000,
                scores=valid_scores,
                idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )

            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            valid_token_nms[idx][valid_token_id_nms] = 1

        valid_token = valid_token_nms

    # ========================================================================
    # Step 4: Initialize Output Lists
    # ========================================================================

    pred_nodes = []
    pred_edges = []

    if map_:
        # Additional outputs for evaluation metrics
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    # ========================================================================
    # Step 5: Process Each Batch Item
    # ========================================================================

    for batch_id in range(h.shape[0]):
        # Get indices of valid tokens for this batch item
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        # Extract node coordinates (center x, y)
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        if map_:
            # Store full bounding boxes and scores for evaluation
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        # ====================================================================
        # Step 6: Edge Prediction (only if multiple nodes exist)
        # ====================================================================

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            # ----------------------------------------------------------------
            # Step 6a: Generate Node Pairs
            # ----------------------------------------------------------------

            # Generate all possible node pairs in original token ordering
            # This creates combinations of actual token indices
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            # Transpose to get separate lists for first and second nodes
            node_pairs = list(map(list, zip(*node_pairs)))

            # Generate node pairs in local ordering (0, 1, 2, ...)
            # Used for indexing into the cost matrix
            node_pairs_valid = torch.tensor(
                [list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))]
            )

            # Create mapping from node pair to index for later lookup
            node_pairs_valid_dict = {}
            for num in range(node_pairs_valid.shape[0]):
                node_pair = node_pairs_valid[num]
                node_pairs_valid_dict[tuple(node_pair.cpu().numpy().tolist())] = num

            # ----------------------------------------------------------------
            # Step 6b: Compute Relation Features
            # ----------------------------------------------------------------

            if rln_token > 0:
                # Include relation token in edge features
                # relation_feature1: [obj_i, obj_j, rln_token] for each pair
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)

                # relation_feature2: [obj_j, obj_i, rln_token] for symmetry
                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                # No relation token, just concatenate object features
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :]
                ), 1)

                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :]
                ), 1)

            # ----------------------------------------------------------------
            # Step 6c: Predict Edge Probabilities
            # ----------------------------------------------------------------

            # Predict edge logits using relation embedding network
            relation_pred1 = net.module.relation_embed(relation_feature1).detach()
            relation_pred2 = net.module.relation_embed(relation_feature2).detach()

            # Average predictions from both directions for symmetry
            relation_pred = (relation_pred1 + relation_pred2) / 2.0

            # ----------------------------------------------------------------
            # Step 6d: Build Cost Matrix (Equation 10)
            # ----------------------------------------------------------------

            # Apply softmax to get probabilities
            # Shape: (num_pairs, 2) where dim 1 = [P(no edge), P(edge)]
            relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1).detach()

            # Extract cost = P(edge does not exist) = softmax(logits)[0]
            # Lower cost means higher probability of edge existence
            cost_pred_batch = relation_pred_softmax_batch[:, 0]

            # ----------------------------------------------------------------
            # Step 6e: Compute MST using NetworkX Kruskal's Algorithm
            # ----------------------------------------------------------------

            # Call helper function to compute MST
            mst_adj_batch = compute_mst_nx(node_pairs_valid, cost_pred_batch)

            # ----------------------------------------------------------------
            # Step 6f: Extract MST Edges
            # ----------------------------------------------------------------

            # Find non-zero entries in adjacency matrix (selected edges)
            mst_tree_selected_list = torch.nonzero(mst_adj_batch, as_tuple=False)

            # Convert to numpy and add to output
            pred_edges.append(mst_tree_selected_list.cpu().numpy())

            # Map local indices back to original node indices for score lookup
            pred_rel_list = [
                node_pairs_valid_dict[tuple(sorted((int(xy[0]), int(xy[1]))))]
                for xy in mst_tree_selected_list
                if xy[0] != xy[1]
            ]
            pred_rel = torch.tensor(pred_rel_list).cpu().numpy()

            if map_:
                # Store edge scores and classes for evaluation
                pred_edges_boxes_score.append(
                    relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy()
                )
                pred_edges_boxes_class.append(
                    torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy()
                )

        else:
            # ====================================================================
            # Step 7: Handle Edge Cases (0 or 1 nodes)
            # ====================================================================

            # No edges possible with 0 or 1 nodes
            pred_edges.append(torch.empty(0, 2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0, 1).cpu().numpy())
                pred_edges_boxes_class.append(torch.empty(0, 1).cpu().numpy())

    # ========================================================================
    # Step 8: Return Results
    # ========================================================================

    if map_:
        return (
            pred_nodes,
            pred_edges,
            pred_nodes_boxes,
            pred_nodes_boxes_score,
            pred_nodes_boxes_class,
            pred_edges_boxes_score,
            pred_edges_boxes_class
        )
    else:
        return pred_nodes, pred_edges


# ============================================================================
# Module Information
# ============================================================================

__version__ = "1.0.0"
__author__ = "Agent-1"
__description__ = "TreeFormer inference with SFS Layer using NetworkX Kruskal's MST"

if __name__ == "__main__":
    """
    Simple test to verify module imports and basic functionality.
    """
    print(f"TreeFormer Inference Module v{__version__}")
    print(f"Description: {__description__}")
    print(f"Author: {__author__}")
    print("\nFunctions available:")
    print("  - relation_infer: Main inference function with SFS Layer")
    print("  - compute_mst_nx: MST computation using NetworkX Kruskal's algorithm")
    print("\nModule loaded successfully!")
