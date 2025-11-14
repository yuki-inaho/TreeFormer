# 損失関数の詳細

## 概要

TreeFormerは複数の損失関数を組み合わせて訓練されます。各損失は異なる側面（ノード検出、分類、エッジ予測）を最適化します。

**実装場所**: `losses_only.py` の `SetCriterion` クラス

## 損失関数の構成

### 重み設定

```yaml
# configs/tree_2D_use_mst_only1.yaml
TRAIN:
  LOSSES: ['boxes', 'class', 'cards', 'nodes', 'edges']
  W_BBOX: 2.0     # バウンディングボックス損失
  W_CLASS: 3.0    # 分類損失
  W_CARD: 1.0     # カーディナリティ損失
  W_NODE: 5.0     # ノード位置損失
  W_EDGE: 4.0     # エッジ損失
```

### 全体損失の計算

```python
total_loss = W_CLASS * loss_class +
             W_NODE * loss_nodes +
             W_BBOX * loss_boxes +
             W_CARD * loss_cardinality +
             W_EDGE * loss_edges
```

## 1. 分類損失 (Classification Loss)

**目的**: ノードの有無を分類

**実装**: `loss_class()` @ `losses_only.py:197-232`

### 詳細

```python
def loss_class(self, outputs, indices):
    # outputs['pred_logits']: [batch, num_queries, 2]
    # indices: Hungarian matchingの結果

    # 重み: [0.2, 0.8] - ノードありを強調
    weight = torch.tensor([0.2, 0.8]).to(device)

    # マッチングされたクエリのインデックス
    idx = self._get_src_permutation_idx(indices)

    # ターゲットラベルの作成
    targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long)
    targets[idx] = 1.0  # マッチングされた位置のみ1

    # Cross Entropy Loss
    loss = F.cross_entropy(
        outputs.permute(0, 2, 1),
        targets,
        weight=weight,
        reduction='mean'
    )
    return loss
```

### 特徴
- **バイナリ分類**: ノードあり(1) vs なし(0)
- **重み付き**: ノードありクラスに0.8の重み（不均衡対策）
- **Hungarian Matching後**: 最適マッチング後のクエリのみ学習

## 2. カーディナリティ損失 (Cardinality Loss)

**目的**: 予測されたノード数がGround Truthと一致するように

**実装**: `loss_cardinality()` @ `losses_only.py:234-261`

### 詳細

```python
def loss_cardinality(self, outputs, indices):
    # 予測されたノード数をカウント
    card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)

    # Ground Truthのノード数
    idx = self._get_src_permutation_idx(indices)
    targets = torch.zeros(outputs[..., 0].shape, dtype=torch.long)
    targets[idx] = 1.0
    tgt_lengths = torch.as_tensor([t.sum() for t in targets])

    # L1 Loss
    loss = F.l1_loss(
        card_pred.float(),
        tgt_lengths.float(),
        reduction='sum'
    ) / (outputs.shape[0] * outputs.shape[1])

    return loss
```

### 特徴
- **補助的損失**: 直接的な勾配は小さいが、統計的な指標として有用
- **正規化**: バッチサイズとクエリ数で正規化

## 3. ノード位置損失 (Node Position Loss)

**目的**: ノードの座標を正確に予測

**実装**: `loss_nodes()` @ `losses_only.py:263-323`

### 詳細

```python
def loss_nodes(self, outputs, targets, indices):
    # outputs: [batch, num_queries, 2]  (x, y)
    # targets: リスト of [num_nodes, 2]

    num_nodes = sum(len(t) for t in targets)

    # Hungarian matchingでマッチングされた予測
    idx = self._get_src_permutation_idx(indices)
    pred_nodes = outputs[idx]

    # マッチングの順序でターゲットを並び替え
    target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

    # L1 Loss
    loss = F.l1_loss(pred_nodes, target_nodes, reduction='none')
    loss = loss.sum() / num_nodes

    return loss
```

### 特徴
- **L1距離**: ロバストな位置推定
- **正規化座標**: [0, 1] 範囲
- **ノード数で正規化**: 異なるサイズのグラフに対応

## 4. バウンディングボックス損失 (Bounding Box Loss)

**目的**: ノードの局所領域を予測（検出の補助）

**実装**: `loss_boxes()` @ `losses_only.py:325-404`

### 詳細

```python
def loss_boxes(self, outputs, targets, indices):
    # outputs: [batch, num_queries, 4]  (x, y, w, h)

    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs[idx]  # 予測

    # ターゲットにwidth/heightを追加
    target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
    width_cat = torch.ones((target_boxes.shape[0], 1)) * (3 / 190)
    height_cat = torch.ones((target_boxes.shape[0], 1)) * (3 / 570)
    target_boxes = torch.cat([target_boxes, width_cat, height_cat], dim=-1)

    # GIoU Loss
    loss = 1 - torch.diag(
        box_ops_2D.generalized_box_iou(
            box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
            box_ops_2D.box_cxcywh_to_xyxy(target_boxes)
        )
    )

    return loss.sum() / num_boxes
```

### 特徴
- **GIoU (Generalized IoU)**: スケール不変な損失
- **固定サイズ**: width=3/190, height=3/570 (データセット依存)
- **補助的**: 主にノード検出の支援

## 5. エッジ損失 (Edge Loss)

**目的**: エッジの存在を予測

**実装**:
- 標準版: `loss_edges()` @ `losses_only.py:406-708`
- MST制約版: `loss_edges_mst_new()` @ `losses_only.py:2362-2360`

### 5.1 標準エッジ損失

```python
def loss_edges(self, h, target_nodes, target_edges, indices, num_edges=500):
    # h: [batch, num_queries, hidden_dim]

    # Object tokens とRelation tokenを分離
    object_token = h[..., :self.obj_token, :]
    relation_token = h[..., self.obj_token:self.obj_token+self.rln_token, :]

    for batch_id in range(batch_size):
        # Positive edges (Ground Truth)
        pos_edge = rearranged_target_edges[batch_id]

        # Negative edges (全組み合わせ - positive)
        full_adj = torch.ones((num_nodes, num_nodes)) - torch.diag(torch.ones(num_nodes))
        full_adj[pos_edge[:, 0], pos_edge[:, 1]] = 0
        full_adj[pos_edge[:, 1], pos_edge[:, 0]] = 0
        neg_edges = torch.nonzero(torch.triu(full_adj))

        # サンプリング
        neg_edges = neg_edges[torch.randperm(neg_edges.shape[0])]

        # 全エッジの特徴量を連結
        all_edges = torch.cat((pos_edge, neg_edges[:take_neg]), 0)

        # Relation token と連結
        relation_feature = torch.cat((
            rearranged_object_token[all_edges[:, 0], :],
            rearranged_object_token[all_edges[:, 1], :],
            relation_token[batch_id, ...].repeat(total_edge, 1)
        ), 1)

        # Edge labels
        edge_labels = torch.cat((
            torch.ones(pos_edge.shape[0]),
            torch.zeros(take_neg)
        ), 0)

    # 全バッチを連結
    relation_feature = torch.cat(relation_features, 0)
    edge_labels = torch.cat(edge_labels, 0)

    # 予測
    relation_pred = self.net.module.relation_embed(relation_feature)

    # NLL Loss
    relation_pred_softmax = F.softmax(relation_pred, dim=-1)
    relation_pred_log = relation_pred_softmax.log()
    loss = nn.NLLLoss()(relation_pred_log, edge_labels)

    return loss / batch_size
```

### 5.2 MST制約付きエッジ損失

**追加の処理** (標準版の後に実行):

```python
# 1. コスト隣接行列の構築
cost_adj_batch = torch.zeros((num_nodes, num_nodes))
relation_pred_softmax = F.softmax(relation_pred_batch, dim=-1)
cost_pred = relation_pred_softmax[:, 0]  # "エッジなし"の確率

for x, y in all_edges:
    cost_adj_batch[x, y] = cost_pred[edge_idx]
    cost_adj_batch[y, x] = cost_pred[edge_idx]

# 2. MSTの計算
from scipy.sparse.csgraph import minimum_spanning_tree
mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy())
mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()

# 3. ラベルの調整
mst_edge_label_batch = torch.ones(pos_edge.shape[0])
for pos_pairs in range(pos_edge.shape[0]):
    x, y = pos_edge[pos_pairs]
    if mst_adj_batch[x, y] == 0:  # MSTに含まれない
        mst_edge_label_batch[pos_pairs] = 0.000001

# 4. 確率分布の再計算
relation_pred_softmax_true = F.softmax(relation_pred_batch, dim=-1)
relation_pred_softmax_true[:, 1] = relation_pred_softmax_true[:, 1] * mst_edge_label_batch
relation_pred_softmax_true[:, 0] = relation_pred_softmax_true[:, 0] + \
                                    relation_pred_softmax_true[:, 1] * (1 - mst_edge_label_batch)

# 5. 損失計算
relation_pred_log_true = relation_pred_softmax_true.log()
nlloss = nn.NLLLoss()(relation_pred_log_true, edge_label_batch)
```

### 特徴

**標準版**:
- Positive/Negative sampling
- バランスの取れた学習
- NLL Loss

**MST制約版**:
- MSTに含まれないエッジを抑制
- 木構造を保証
- 確率分布を動的に調整

## エッジサンプリング戦略

### Positive Edges
- Ground Truthから直接取得
- 全て使用（サンプリングなし）

### Negative Edges
- 全ノードペア - Positive edges
- ランダムサンプリング
- シャッフル（無向グラフのため）

### バランス
```python
# Negativeサンプル数の決定
take_neg = neg_edges.shape[0]  # 全Negative edgesを使用
total_edge = pos_edge.shape[0] + take_neg
```

## 損失の流れ

```
Forward Pass:
    Image → Encoder → Decoder → Predictions
                                    ↓
                        ┌───────────┼───────────┐
                        ↓           ↓           ↓
                    Class       Nodes       Edges
                        ↓           ↓           ↓
            Hungarian Matching (indices)
                        ↓           ↓           ↓
                   loss_class  loss_nodes  loss_edges
                        ↓           ↓           ↓
                        └───────────┼───────────┘
                                    ↓
                              Total Loss

MST Constraint (if enabled):
    loss_edges → MST Calculation → Label Adjustment → Modified Loss
```

## コード内の対応表

| 損失関数 | メソッド | ファイル | 行番号 | 重み |
|---------|---------|---------|--------|------|
| 分類損失 | `loss_class()` | `losses_only.py` | 197-232 | W_CLASS: 3.0 |
| カーディナリティ | `loss_cardinality()` | `losses_only.py` | 234-261 | W_CARD: 1.0 |
| ノード位置 | `loss_nodes()` | `losses_only.py` | 263-323 | W_NODE: 5.0 |
| バウンディングボックス | `loss_boxes()` | `losses_only.py` | 325-404 | W_BBOX: 2.0 |
| エッジ(標準) | `loss_edges()` | `losses_only.py` | 406-708 | W_EDGE: 4.0 |
| エッジ(MST) | `loss_edges_mst_new()` | `losses_only.py` | 2362- | W_EDGE: 4.0 |

## 損失のバランス調整

### 推奨設定 (論文設定)

```yaml
W_BBOX: 2.0    # バウンディングボックス - 補助的
W_CLASS: 3.0   # 分類 - 重要
W_CARD: 1.0    # カーディナリティ - 補助的
W_NODE: 5.0    # ノード位置 - 最重要
W_EDGE: 4.0    # エッジ - 非常に重要
```

### 調整のヒント

1. **W_NODE**: ノード検出の精度に直接影響
2. **W_EDGE**: グラフ構造の品質に影響
3. **W_CLASS**: False positiveの抑制に重要
4. **W_BBOX, W_CARD**: 補助的、大きくしすぎない

## 訓練時の損失の推移

### 典型的な推移 (100 epochs)

```
Epoch   Total   Class   Node    Edge    Box     Card
1       15.2    2.8     8.1     3.5     0.6     0.2
10       8.4    1.2     4.3     2.3     0.4     0.2
50       3.1    0.3     1.5     1.0     0.2     0.1
100      1.8    0.2     0.8     0.6     0.1     0.1
```

## まとめ

TreeFormerの損失関数は:
1. **多面的**: ノード、エッジ、分類を個別に最適化
2. **バランス**: 重み付けで各要素の重要性を調整
3. **制約付き**: MST制約で木構造を保証
4. **ロバスト**: L1損失とGIoUでスケール不変性を確保
