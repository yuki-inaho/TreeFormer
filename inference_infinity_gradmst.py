"""
Inference Module: Unconstrained Baseline (Without MST Constraint)

This module implements the unconstrained baseline for relation inference in scene graphs,
as described in the TreeFormer paper Section 5.3 "Unconstrained [55]".

Unlike MST-constrained approaches that enforce a tree structure, this baseline uses simple
threshold-based edge selection where an edge (i,j) exists if:
    p(exist | i,j) > p(non-exist | i,j)

This is the simplest approach using argmax classification without any structural constraints.

Reference:
    Paper Section 5.3: Unconstrained baseline method [55]
    Base implementation: epoch.py relation_infer function

Key Differences from MST Versions:
    - No Minimum Spanning Tree computation
    - No NetworkX dependency
    - Simple binary classification per edge
    - May produce disconnected graphs or cycles
    - Faster inference (no graph algorithm overhead)

Threshold Selection:
    Edges are selected using argmax over the existence probability:
        E = {(i,j) | argmax(p_ij) == 1}
    where p_ij = [p(non-exist), p(exist)] is the 2-dimensional logit output.

Author: TreeFormer Project
Date: 2025-11-14
"""

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import itertools
import numpy as np
from typing import Tuple, List, Union, Optional


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
    Infer scene graph relations using unconstrained threshold-based edge selection.

    This function implements the unconstrained baseline that predicts edges by simple
    binary classification without any structural constraints (no MST). Each potential
    edge (i,j) is classified independently as exist/non-exist based on threshold.

    Args:
        h (torch.Tensor): Hidden feature tensor from transformer encoder.
            Shape: [batch_size, num_tokens, hidden_dim]
            Contains both object tokens and optional relation token.

        out (dict): Model output dictionary containing:
            - 'pred_logits': Node existence logits, shape [batch, num_tokens, 2]
                Used to determine which tokens represent valid objects.
            - 'pred_nodes': Predicted node bounding boxes, shape [batch, num_tokens, 4]
                Format: [center_x, center_y, width, height] in normalized coordinates.

        net (torch.nn.Module): The neural network model.
            Must have 'module.relation_embed' attribute for relation prediction.

        obj_token (int): Number of object tokens in the hidden features.
            Typically 20 for standard scene graph datasets.

        rln_token (int): Number of relation tokens (0 or 1).
            If > 0, the last token in h is treated as a global relation token.

        nms (bool, optional): Whether to apply Non-Maximum Suppression on detected objects.
            Default: False. If True, removes duplicate/overlapping object detections.

        map_ (bool, optional): Whether to return additional mAP evaluation data.
            Default: False. If True, returns confidence scores and class predictions.

    Returns:
        If map_ is False:
            pred_nodes (List[torch.Tensor]): List of predicted node coordinates per batch.
                Each element shape: [num_valid_nodes, 2], normalized [x, y] coordinates.

            pred_edges (List[np.ndarray]): List of predicted edges per batch.
                Each element shape: [num_edges, 2], containing node indices (i, j).
                Edges selected by threshold: argmax(relation_logits) == 1.

        If map_ is True:
            Returns 7-tuple with additional evaluation data:
            - pred_nodes: Node coordinates
            - pred_edges: Edge indices
            - pred_nodes_boxes: Full bounding boxes [x, y, w, h]
            - pred_nodes_boxes_score: Node confidence scores
            - pred_nodes_boxes_class: Node class labels
            - pred_edges_boxes_score: Edge confidence scores
            - pred_edges_boxes_class: Edge class labels

    Algorithm:
        1. Extract object tokens and relation token from hidden features h
        2. Determine valid objects using argmax on pred_logits
        3. Optional: Apply NMS to remove duplicate detections
        4. For each batch:
            a. Get indices of valid nodes
            b. Generate all possible node pairs (combinations)
            c. For each pair (i,j):
                - Concatenate features: [obj_i, obj_j, rln_token]
                - Predict relation in both directions
                - Average bidirectional predictions
            d. Select edges using threshold: argmax(relation_pred) == 1
        5. Return predicted nodes and edges

    Edge Selection (Unconstrained):
        Unlike MST-based methods, this baseline uses simple binary classification:
            E = {(i,j) | p(exist | i,j) > p(non-exist | i,j)}

        Implementation: torch.argmax(relation_pred, dim=-1)
        - If argmax returns 1 → edge exists
        - If argmax returns 0 → no edge

        This can produce:
        - Disconnected graphs (isolated nodes)
        - Cycles (no tree constraint)
        - Variable number of edges (not necessarily n-1)

    Example:
        >>> h = torch.randn(2, 21, 256)  # 2 batches, 20 obj + 1 rln token
        >>> out = {
        ...     'pred_logits': torch.randn(2, 20, 2),
        ...     'pred_nodes': torch.rand(2, 20, 4)
        ... }
        >>> nodes, edges = relation_infer(h, out, model, 20, 1)
        >>> print(nodes[0].shape)  # e.g., torch.Size([7, 2]) for 7 valid nodes
        >>> print(edges[0].shape)  # e.g., (6, 2) for 6 predicted edges

    Notes:
        - No graph structure constraints enforced
        - Faster than MST methods (no graph algorithm)
        - May have lower recall for weakly-connected graphs
        - Suitable as baseline for comparison
        - NMS can reduce false positives from duplicate detections

    Raises:
        AttributeError: If net does not have 'module.relation_embed' attribute.
        RuntimeError: If tensor dimensions mismatch expectations.
    """
    # Extract object tokens (all tokens except the last one)
    # Shape: [batch_size, obj_token, hidden_dim]
    object_token = h[..., :obj_token, :]

    # Extract relation token if present (last token)
    # Shape: [batch_size, rln_token, hidden_dim]
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    # Determine valid tokens using argmax on object existence logits
    # Shape: [batch_size, obj_token]
    # Returns 0 for non-existent objects, 1 for existent objects
    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    # Apply Non-Maximum Suppression if requested
    # Removes duplicate/overlapping object detections based on IoU threshold
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(
            zip(valid_token, out['pred_logits'], out['pred_nodes'])
        ):
            # Get indices of valid tokens (where token == 1)
            valid_token_id = torch.nonzero(token).squeeze(1)

            # Extract logits and boxes for valid tokens only
            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]

            # Compute confidence scores using softmax
            # Shape: [num_valid_tokens]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # Convert center format to corner format for NMS
            # NMS requires format: [x1, y1, x2, y2] where x1 < x2 and y1 < y2
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            # Apply batched NMS with IoU threshold 0.90
            # Scale coordinates by 1000 for numerical stability
            ids2keep = batched_nms(
                boxes=valid_nodes * 1000,
                scores=valid_scores,
                idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )

            # Update valid token mask with NMS results
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            valid_token_nms[idx][valid_token_id_nms] = 1

        valid_token = valid_token_nms

    # Initialize output lists
    pred_nodes = []
    pred_edges = []

    # Initialize additional outputs for mAP evaluation if requested
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    # Process each batch independently
    for batch_id in range(h.shape[0]):
        # Get indices of valid tokens (detected objects) for this batch
        # Shape: [num_valid_nodes]
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        # Extract predicted node coordinates (only center x, y)
        # Append to pred_nodes list
        # Shape: [num_valid_nodes, 2]
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        # Store additional node information for mAP evaluation
        if map_:
            # Full bounding boxes [x, y, w, h]
            pred_nodes_boxes.append(
                out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy()
            )

            # Node confidence scores (probability of existence)
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )

            # Node class labels (1 for valid objects)
            pred_nodes_boxes_class.append(
                valid_token[batch_id, node_id].cpu().numpy()
            )

        # Only process edges if we have at least 2 valid nodes
        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            # Generate all possible node pairs using combinations
            # Example: nodes [2, 6, 8] → pairs [(2,6), (2,8), (6,8)]
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]

            # Transpose to get separate lists for source and target nodes
            # Format: [[src_1, src_2, ...], [tgt_1, tgt_2, ...]]
            node_pairs = list(map(list, zip(*node_pairs)))

            # Generate node pairs in valid token order (0-indexed within valid nodes)
            # This is used for indexing into the predicted edges
            # Shape: [num_pairs, 2]
            node_pairs_valid = torch.tensor([
                list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))
            ])

            # Construct relation features for edge prediction
            # Concatenate: [object_i, object_j, relation_token]
            if rln_token > 0:
                # Forward direction: (i → j)
                # Shape: [num_pairs, hidden_dim * 3]
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)

                # Backward direction: (j → i)
                # Shape: [num_pairs, hidden_dim * 3]
                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                # No relation token, only concatenate object features
                # Shape: [num_pairs, hidden_dim * 2]
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :]
                ), 1)

                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :]
                ), 1)

            # Predict relation logits for both directions
            # Shape: [num_pairs, 2] where 2 = [non-exist, exist]
            relation_pred1 = net.module.relation_embed(relation_feature1).detach()
            relation_pred2 = net.module.relation_embed(relation_feature2).detach()

            # Average bidirectional predictions for symmetric relation scores
            # This improves robustness by considering both orderings
            # Shape: [num_pairs, 2]
            relation_pred = (relation_pred1 + relation_pred2) / 2.0

            # UNCONSTRAINED EDGE SELECTION (KEY DIFFERENCE FROM MST)
            # Select edges using simple threshold: argmax(relation_pred) == 1
            # Returns indices where argmax is 1 (edge exists)
            # Shape: [num_selected_edges]
            pred_rel = torch.nonzero(torch.argmax(relation_pred, -1)).squeeze(1).cpu().numpy()

            # Extract edge pairs corresponding to selected relations
            # Shape: [num_selected_edges, 2]
            pred_edges.append(node_pairs_valid[pred_rel].cpu().numpy())

            # Store additional edge information for mAP evaluation
            if map_:
                # Edge confidence scores (probability of existence)
                pred_edges_boxes_score.append(
                    relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy()
                )

                # Edge class labels (1 for existing edges)
                pred_edges_boxes_class.append(
                    torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy()
                )
        else:
            # No valid node pairs, append empty edge list
            pred_edges.append(torch.empty(0, 2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0, 1).cpu().numpy())
                pred_edges_boxes_class.append(torch.empty(0, 1).cpu().numpy())

    # Return results based on map_ flag
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


# Module metadata
__version__ = "1.0.0"
__author__ = "TreeFormer Project"
__description__ = "Unconstrained baseline inference without MST constraint"

# Export public API
__all__ = ['relation_infer']


# Usage example (for documentation purposes)
if __name__ == "__main__":
    """
    Example usage of the unconstrained relation inference.

    This demonstrates how to use the relation_infer function for scene graph
    prediction without MST constraints.
    """
    print("TreeFormer Unconstrained Inference Module")
    print("=" * 50)
    print(__description__)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print("\nThis module provides threshold-based edge selection without")
    print("structural constraints, serving as a baseline for comparison.")
    print("\nKey characteristics:")
    print("  - No MST computation required")
    print("  - Simple binary classification per edge")
    print("  - Faster inference than graph-constrained methods")
    print("  - May produce disconnected or cyclic graphs")
    print("\nFor actual usage, import and call relation_infer() with your")
    print("model outputs and configuration.")
