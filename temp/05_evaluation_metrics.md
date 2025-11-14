# 評価メトリクス

## 概要

TreeFormerは複数のメトリクスで評価されます。各メトリクスは異なる側面（ノード検出、エッジ予測、トポロジー）を測定します。

**評価スクリプト**: `valid_smd_guyot_nx.py`

## メトリクスの種類

| メトリクス | 目的 | 範囲 | 理想値 |
|----------|------|------|--------|
| SMD | 骨格マッチング距離 | [0, ∞) | 0に近いほど良い |
| TOPO | トポロジー保存 | [0, 1] | 1に近いほど良い |
| mAP | ノード検出精度 | [0, 1] | 1に近いほど良い |
| Precision/Recall | エッジ精度 | [0, 1] | 1に近いほど良い |

## 1. SMD (Skeleton Matching Distance)

### 概要

**SMD**は予測されたスケルトンとGround Truthスケルトン間の距離を測定します。

### 計算方法

**実装**: `metric_smd.py`

```python
def compute_smd(pred_points, pred_edges, gt_points, gt_edges, img_size=512):
    """
    Args:
        pred_points: [N, 2] 予測ノード座標
        pred_edges: [E, 2] 予測エッジ
        gt_points: [M, 2] Ground Truthノード座標
        gt_edges: [F, 2] Ground Truthエッジ
        img_size: 画像サイズ

    Returns:
        smd: float スケルトンマッチング距離
    """

    # 1. 座標を画像空間にスケール
    pred_points_scaled = pred_points * img_size
    gt_points_scaled = gt_points * img_size

    # 2. エッジをポリラインとして表現
    pred_polylines = create_polylines(pred_points_scaled, pred_edges)
    gt_polylines = create_polylines(gt_points_scaled, gt_edges)

    # 3. 各予測ポイントから最近傍GTポリラインへの距離
    dist_pred_to_gt = []
    for pred_line in pred_polylines:
        min_dist = float('inf')
        for gt_line in gt_polylines:
            dist = polyline_distance(pred_line, gt_line)
            min_dist = min(min_dist, dist)
        dist_pred_to_gt.append(min_dist)

    # 4. 各GTポイントから最近傍予測ポリラインへの距離
    dist_gt_to_pred = []
    for gt_line in gt_polylines:
        min_dist = float('inf')
        for pred_line in pred_polylines:
            dist = polyline_distance(gt_line, pred_line)
            min_dist = min(min_dist, dist)
        dist_gt_to_pred.append(min_dist)

    # 5. 平均距離
    smd = (sum(dist_pred_to_gt) + sum(dist_gt_to_pred)) / \
          (len(dist_pred_to_gt) + len(dist_gt_to_pred))

    return smd
```

### 特徴

- **双方向**: 予測→GT と GT→予測 の両方向を考慮
- **ポリライン**: エッジを連続的な線として扱う
- **スケール**: ピクセル単位で測定

### 解釈

```
SMD = 0    : 完全一致
SMD < 5px  : 非常に良い
SMD < 10px : 良い
SMD > 20px : 改善が必要
```

## 2. TOPO (Topology Metric)

### 概要

**TOPO**はグラフのトポロジー構造（分岐、接続）が保存されているかを測定します。

### 理論

TOPOメトリクスは以下の論文に基づきます:
- "Topology-Preserving Road Network Extraction"
- "Marbles on Holes" マッチング

### 計算方法

**実装**: `metric_topo/topo.py`

```python
def compute_topo(pred_graph, gt_graph, threshold=0.00015):
    """
    Args:
        pred_graph: 予測されたグラフ (nodes, edges)
        gt_graph: Ground Truthグラフ
        threshold: マッチング閾値 (lat/lon単位)

    Returns:
        precision: float [0, 1]
        recall: float [0, 1]
    """

    # 1. グラフをRoadGraphオブジェクトに変換
    pred_road_graph = create_graph(pred_graph)
    gt_road_graph = create_graph(gt_graph)

    # 2. 開始点を生成 (エッジに沿って等間隔)
    osm_list = TOPOGenerateStartingPoints(
        gt_road_graph,
        density=0.00050,  # ~50m間隔
        region=region
    )

    # 3. 予測グラフとGTグラフをペアリング
    gps_list = TOPOGeneratePairs(
        pred_road_graph,
        gt_road_graph,
        osm_list,
        threshold=threshold
    )

    # 4. 各ペアでTOPO Walkを実行
    topo_result = TOPOWithPairs(
        pred_road_graph,
        gt_road_graph,
        gps_list,
        osm_list,
        r=0.00300,  # ~300m 探索半径
        step=0.00005,
        threshold=threshold
    )

    precision, recall = topo_result
    return precision, recall
```

### TOPO Walk

**コンセプト**: "Marbles on Holes"

1. **開始点**: エッジ上のサンプル点から開始
2. **探索**: 半径rの範囲内でグラフをたどる
3. **マッチング**: 予測の探索点とGTの探索点をマッチング
4. **精度計算**:
   - Precision = マッチした予測点 / 全予測点
   - Recall = マッチしたGT点 / 全GT点

```python
def TOPOWalk(graph, start_node, step=0.00005, r=0.00300):
    """
    グラフ上をランダムウォーク
    """
    visited_points = []
    current_pos = graph.nodes[start_node]
    total_distance = 0

    while total_distance < r:
        # 次のノードを選択
        neighbors = graph.nodeLink[current_node]
        next_node = random.choice(neighbors)

        # エッジに沿って進む
        edge_vector = graph.nodes[next_node] - current_pos
        edge_length = np.linalg.norm(edge_vector)

        if total_distance + edge_length > r:
            # 半径rに到達
            remaining = r - total_distance
            current_pos = current_pos + edge_vector * (remaining / edge_length)
            visited_points.append(current_pos)
            break
        else:
            # stepごとにサンプル
            num_steps = int(edge_length / step)
            for i in range(num_steps):
                pos = current_pos + edge_vector * (i * step / edge_length)
                visited_points.append(pos)
                total_distance += step

            current_node = next_node
            current_pos = graph.nodes[current_node]

    return visited_points
```

### 特徴

- **局所的**: 各開始点から局所的な構造を評価
- **ロバスト**: 小さなノイズに強い
- **トポロジー重視**: 接続関係を重視

### 解釈

```
Precision/Recall > 0.9 : 非常に良い
Precision/Recall > 0.7 : 良い
Precision/Recall > 0.5 : 改善が必要
Precision/Recall < 0.3 : 不十分
```

## 3. mAP (Mean Average Precision)

### 概要

**mAP**はノード検出の精度を測定します。Object Detectionの標準メトリクスです。

### 計算方法

**実装**: `metric_map.py`

```python
def compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Args:
        pred_boxes: [N, 4] 予測バウンディングボックス (x, y, w, h)
        pred_scores: [N] 予測スコア
        gt_boxes: [M, 4] Ground Truthボックス
        iou_threshold: IoU閾値

    Returns:
        ap: float Average Precision
    """

    # 1. スコアで降順ソート
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # 2. 各予測とGTのマッチング
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        max_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        if max_iou >= iou_threshold:
            tp[i] = 1
            matched_gt.add(max_idx)
        else:
            fp[i] = 1

    # 3. Precision-Recall曲線
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)

    recalls = cumsum_tp / len(gt_boxes)
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp)

    # 4. Average Precision (台形近似)
    ap = 0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]

    return ap
```

### mAP@IoU

複数のIoU閾値でAPを計算し、平均を取る:

```python
iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
aps = [compute_map(pred_boxes, pred_scores, gt_boxes, iou) for iou in iou_thresholds]
map_score = np.mean(aps)
```

### 解釈

```
mAP > 0.8 : 非常に良い
mAP > 0.6 : 良い
mAP > 0.4 : 改善が必要
mAP < 0.2 : 不十分
```

## 4. Edge Precision/Recall

### 概要

エッジ予測の精度を直接測定します。

### 計算方法

```python
def compute_edge_metrics(pred_edges, gt_edges, pred_nodes, gt_nodes, threshold=5):
    """
    Args:
        pred_edges: [E1, 2] 予測エッジのインデックス
        gt_edges: [E2, 2] Ground Truthエッジ
        pred_nodes: [N1, 2] 予測ノード座標
        gt_nodes: [N2, 2] GTノード座標
        threshold: ノードマッチング閾値 (ピクセル)

    Returns:
        precision: float
        recall: float
        f1: float
    """

    # 1. ノードマッチングを計算
    node_matches = match_nodes(pred_nodes, gt_nodes, threshold)
    # node_matches[pred_idx] = gt_idx or -1

    # 2. エッジをマッチング
    matched_edges = []
    for pred_edge in pred_edges:
        pred_n1, pred_n2 = pred_edge
        gt_n1 = node_matches[pred_n1]
        gt_n2 = node_matches[pred_n2]

        if gt_n1 == -1 or gt_n2 == -1:
            continue

        # GTエッジに存在するか確認
        if (gt_n1, gt_n2) in gt_edges or (gt_n2, gt_n1) in gt_edges:
            matched_edges.append(pred_edge)

    # 3. Precision/Recall
    tp = len(matched_edges)
    fp = len(pred_edges) - tp
    fn = len(gt_edges) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
```

### 特徴

- **ノードベース**: まずノードをマッチング、次にエッジ
- **双方向**: 無向グラフとして扱う
- **閾値依存**: ノードマッチング閾値に敏感

## 5. NetworkX ベースのメトリクス

### グラフ統計

```python
import networkx as nx

def compute_graph_statistics(pred_graph, gt_graph):
    """
    グラフの統計的特性を比較
    """
    G_pred = nx.Graph()
    G_pred.add_edges_from(pred_edges)

    G_gt = nx.Graph()
    G_gt.add_edges_from(gt_edges)

    stats = {
        'num_nodes_pred': G_pred.number_of_nodes(),
        'num_nodes_gt': G_gt.number_of_nodes(),
        'num_edges_pred': G_pred.number_of_edges(),
        'num_edges_gt': G_gt.number_of_edges(),
        'avg_degree_pred': np.mean([d for n, d in G_pred.degree()]),
        'avg_degree_gt': np.mean([d for n, d in G_gt.degree()]),
        'is_tree_pred': nx.is_tree(G_pred),
        'is_tree_gt': nx.is_tree(G_gt),
        'num_connected_components_pred': nx.number_connected_components(G_pred),
        'num_connected_components_gt': nx.number_connected_components(G_gt),
    }

    return stats
```

### トポロジカル距離

```python
def graph_edit_distance(pred_graph, gt_graph):
    """
    グラフ編集距離 (Graph Edit Distance)
    """
    G_pred = nx.Graph()
    G_pred.add_edges_from(pred_edges)

    G_gt = nx.Graph()
    G_gt.add_edges_from(gt_edges)

    # 近似アルゴリズム (完全計算は指数時間)
    ged = nx.graph_edit_distance(G_pred, G_gt, timeout=10)

    return ged
```

## 評価の実行

### コマンド

```bash
python valid_smd_guyot_nx.py \
  --config configs/tree_2D_use_mst_only1.yaml \
  --checkpoint trained_weights/checkpoint_best.pkl
```

### 出力

```
Evaluating on test set...
Sample 1/100: SMD=3.24, TOPO_P=0.92, TOPO_R=0.89, mAP=0.85
Sample 2/100: SMD=4.11, TOPO_P=0.88, TOPO_R=0.91, mAP=0.83
...
Sample 100/100: SMD=3.87, TOPO_P=0.90, TOPO_R=0.87, mAP=0.84

Average Results:
  SMD: 3.56 ± 1.23
  TOPO Precision: 0.904 ± 0.045
  TOPO Recall: 0.891 ± 0.038
  mAP@0.5: 0.847 ± 0.062
  Edge Precision: 0.873 ± 0.071
  Edge Recall: 0.865 ± 0.068
```

## ベンチマーク結果

### Guyot Dataset

| モデル | SMD↓ | TOPO-P↑ | TOPO-R↑ | mAP↑ |
|-------|------|---------|---------|------|
| TreeFormer (MST) | **3.56** | **0.904** | **0.891** | **0.847** |
| TreeFormer (no MST) | 4.23 | 0.867 | 0.854 | 0.821 |
| Baseline DETR | 5.91 | 0.782 | 0.795 | 0.763 |

### 分析

**MST制約の効果**:
- SMD: 15.8% 改善
- TOPO: 4-5% 改善
- mAP: 3.2% 改善

## 可視化

### 予測結果の可視化

```python
import matplotlib.pyplot as plt

def visualize_prediction(image, pred_nodes, pred_edges, gt_nodes, gt_edges, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')

    # 2. Prediction
    axes[1].imshow(image, alpha=0.5)
    for edge in pred_edges:
        n1, n2 = edge
        axes[1].plot([pred_nodes[n1, 0], pred_nodes[n2, 0]],
                     [pred_nodes[n1, 1], pred_nodes[n2, 1]], 'r-', linewidth=2)
    axes[1].scatter(pred_nodes[:, 0], pred_nodes[:, 1], c='red', s=50)
    axes[1].set_title(f'Prediction (SMD={metrics["smd"]:.2f})')

    # 3. Ground Truth
    axes[2].imshow(image, alpha=0.5)
    for edge in gt_edges:
        n1, n2 = edge
        axes[2].plot([gt_nodes[n1, 0], gt_nodes[n2, 0]],
                     [gt_nodes[n1, 1], gt_nodes[n2, 1]], 'g-', linewidth=2)
    axes[2].scatter(gt_nodes[:, 0], gt_nodes[:, 1], c='green', s=50)
    axes[2].set_title('Ground Truth')

    plt.tight_layout()
    plt.savefig('prediction_vis.png')
```

## まとめ

TreeFormerの評価は多面的です:
1. **SMD**: 幾何学的精度
2. **TOPO**: トポロジー保存
3. **mAP**: ノード検出
4. **Precision/Recall**: エッジ精度
5. **グラフ統計**: 構造的特性

全てのメトリクスが重要であり、総合的な評価が必要です。
