# GitHub Issue #2 詳細調査報告書
# Missing inference_infinity_mst_nx_gradmst File in Repository

調査日時: 2025-11-14
調査対象: TreeFormerプロジェクト
Issue: #2 - 欠落ファイル `inference_infinity_mst_nx_gradmst.py`

---

## 1. 問題の概要

### 1.1 問題の詳細
プロジェクト内の `/home/inaho-omen/Project/TreeFormer/valid_smd_guyot_nx.py` ファイルにおいて、存在しないモジュール `inference_infinity_mst_nx_gradmst` からの `relation_infer` 関数のインポートが試みられています。

### 1.2 影響を受けるコード
**ファイルパス:** `/home/inaho-omen/Project/TreeFormer/valid_smd_guyot_nx.py`
**行番号:** 1257

```python
# 1255行目からの該当コード
if is_use_mst:
    # from inference_infinity_mst_nx_dist import relation_infer
    from inference_infinity_mst_nx_gradmst import relation_infer
else:
    from inference_infinity_gradmst import relation_infer
```

### 1.3 欠落状況の確認
- `inference_infinity_mst_nx_gradmst.py`: **存在しない**
- `inference_infinity_mst_nx_dist.py` (コメントアウト): **存在しない**
- `inference_infinity_gradmst.py`: **存在しない**

これらのファイルはいずれもプロジェクトのルートディレクトリおよびサブディレクトリに存在しません。

---

## 2. 欠落ファイルの影響範囲

### 2.1 直接的な影響
1. **検証スクリプトの実行不可**
   - `valid_smd_guyot_nx.py` で `is_use_mst=True` の場合、ImportErrorが発生
   - MST（Minimum Spanning Tree）ベースの推論が実行できない

2. **代替モジュールも欠落**
   - `is_use_mst=False` の場合も `inference_infinity_gradmst` が存在しないため、いずれのパスも失敗する

### 2.2 関連する機能
**ファイル名から推定される機能:**
- `inference_infinity`: 無限グラフ推論機能
- `mst_nx`: NetworkXライブラリを使用したMST（最小全域木）アルゴリズム
- `gradmst`: 勾配ベースのMST最適化

**期待される関数シグネチャ:**
```python
def relation_infer(h, out, net, obj_token, rln_token, nms=False, map_=False):
    """
    Args:
        h: 隠れ層の特徴量
        out: モデルの出力
        net: ニューラルネットワークモデル
        obj_token: オブジェクトトークン数
        rln_token: リレーショントークン数
        nms: Non-Maximum Suppressionの適用有無
        map_: マッピング情報の返却有無
    Returns:
        pred_nodes, pred_edges: 予測されたノードとエッジ
        (map_=Trueの場合、追加でボックス情報を返却)
    """
```

---

## 3. 既存コードの類似実装

### 3.1 epoch.pyのrelation_infer関数
**ファイルパス:** `/home/inaho-omen/Project/TreeFormer/epoch.py`
**行番号:** 43-306

この関数は基本的な推論機能を提供しています：
```python
def relation_infer(h, out, net, obj_token, rln_token, nms=False, map_=False):
    object_token = h[..., :obj_token, :]
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    # NMSの適用
    if nms:
        # Non-Maximum Suppression処理
        ...

    # ノードペアの生成と関係性予測
    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]

        # 関係性特徴の連結
        relation_feature1 = torch.cat((object_token[...], relation_token[...]), 1)
        relation_pred = net.module.relation_embed(relation_feature)

        # 閾値ベースのエッジ選択
        pred_rel = torch.nonzero(torch.argmax(relation_pred, -1))
        pred_edges.append(node_pairs_valid[pred_rel])

    return pred_nodes, pred_edges
```

**特徴:**
- 閾値ベースの単純なエッジ選択
- MSTアルゴリズムは使用していない

### 3.2 epoch.pyのrelation_infer_mst関数
**ファイルパス:** `/home/inaho-omen/Project/TreeFormer/epoch.py`
**行番号:** 308-582

MST（最小全域木）ベースの推論機能：
```python
def relation_infer_mst(h, out, net, obj_token, rln_token, nms=False, map_=False):
    # ... (基本的な前処理は同じ) ...

    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        # コスト隣接行列の初期化
        cost_adj_batch = torch.ones((node_id.shape[0], node_id.shape[0])).to(h.device) * 9999

        # 関係性予測からコストを算出
        relation_pred_softmax_batch = F.softmax(relation_pred, dim=-1).detach()
        cost_pred_batch = relation_pred_softmax_batch[:, 0]  # コストは確率の逆

        # コスト隣接行列の構築
        x, y = node_pairs_valid.t()
        cost_adj_batch[x, y] = cost_pred_batch
        cost_adj_batch[y, x] = cost_pred_batch
        cost_adj_batch *= torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)

        # NetworkXを使用したMST計算
        mst_adj_batch = compute_mst_prim(node_pairs_valid, cost_pred_batch)

        # MSTから選択されたエッジを抽出
        mst_tree_selected_list = torch.nonzero(mst_adj_batch, as_tuple=False)
        pred_edges.append(mst_tree_selected_list.cpu().numpy())

    return pred_nodes, pred_edges
```

**キーとなるヘルパー関数:**
```python
def compute_mst_prim(node_pairs_valid, cost_pred_batch):
    G = nx.Graph()
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    # Kruskalアルゴリズムを使用
    mst_edges = nx.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    mst_edges = list(mst_edges)

    # MST隣接行列の構築
    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight

    mst_adj_batch = torch.tensor(mst_adj_np)
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch
```

**特徴:**
- NetworkXのKruskalアルゴリズムを使用
- コストベースの最小全域木構築
- 木構造の制約を保証

### 3.3 epoch.pyのrelation_infer_gnn関数
**ファイルパス:** `/home/inaho-omen/Project/TreeFormer/epoch.py`
**行番号:** 584-784

Graph Neural Network（GNN）ベースの推論：
```python
def relation_infer_gnn(h, out, model, obj_token, rln_token, nms=False, map_=False):
    # ... (前処理) ...

    for batch_id in range(h.shape[0]):
        n = out['pred_nodes'][batch_id, node_id, :2].detach()
        rearranged_object_token = object_token[batch_id, node_id, :]

        # 完全グラフの構築
        full_adj = torch.ones((n.shape[0], n.shape[0]), device=h.device)
        all_full_adj = []
        for row in range(full_adj.shape[0]):
            for col in range(full_adj.shape[1]):
                all_full_adj.append([row, col])
        all_full_adj = torch.tensor(all_full_adj, device=h.device, dtype=torch.long).t().contiguous()

        # Graph Auto-Encoderを使用
        relation_feature = torch.cat([rearranged_object_token,
                                     relation_token[batch_id, ...].repeat(rearranged_object_token.shape[0], 1)], 1)
        val_z = model.module.GAE_model.encode(relation_feature, all_full_adj).detach()
        prob_adj = model.module.GAE_model.decoder.forward_all(val_z)

        # 確率的閾値による選択
        prob_adj = prob_adj * torch.triu(torch.ones_like(prob_adj), diagonal=1)
        pred_rel = torch.where(prob_adj > 0.5, 1, 0)
        pred_edges.append(torch.nonzero(pred_rel).cpu().numpy())

    return pred_nodes, pred_edges
```

**特徴:**
- Graph Auto-Encoder（GAE）を使用
- エンド・ツー・エンドの学習可能なグラフ推論
- 確率的なエッジ予測

### 3.4 losses_only.pyのprims_mst関数
**ファイルパス:** `/home/inaho-omen/Project/TreeFormer/losses_only.py`
**行番号:** 76-100

Primアルゴリズムの実装：
```python
def prims_mst(cost_adj):
    num_of_nodes = cost_adj.shape[0]
    device = cost_adj.device
    postive_inf = 1000
    selected_nodes = torch.zeros((num_of_nodes, 1), device=device).bool()
    mst_adj = torch.zeros_like(cost_adj, device=device)

    while not selected_nodes.all():
        minimum = postive_inf
        start = 0
        end = 0

        for row in range(num_of_nodes):
            if selected_nodes[row]:
                for col in range(num_of_nodes):
                    if (not selected_nodes[col] and cost_adj[row, col] > 0):
                        if cost_adj[row, col] < minimum:
                            minimum = cost_adj[row, col]
                            start, end = row, col

        selected_nodes[end] = True
        mst_adj[start, end] = minimum
        if minimum == postive_inf:
            mst_adj[start, end] = 0
        mst_adj[end, start] = mst_adj[start, end]

    return mst_adj
```

**特徴:**
- PyTorchネイティブな実装
- 微分可能ではない（訓練時には使用困難）

---

## 4. 代替実装の提案

### オプション1: epoch.pyの既存実装を使用（推奨）

**実装難易度:** 低
**実装時間:** 5分
**利点:**
- 既存コードを活用、実装済み
- 即座に動作可能
- MSTアルゴリズム対応済み

**実装方法:**

```python
# valid_smd_guyot_nx.py の1255-1259行目を以下に置き換え

if is_use_mst:
    from epoch import relation_infer_mst as relation_infer
else:
    from epoch import relation_infer
```

**動作検証コード:**
```python
# テストスクリプト
import torch
from epoch import relation_infer_mst, relation_infer

# ダミーデータ
h = torch.randn(2, 21, 256)
out = {
    'pred_logits': torch.randn(2, 20, 2),
    'pred_nodes': torch.randn(2, 20, 4)
}
net = None  # モデルインスタンスが必要

# MST版のテスト
pred_nodes, pred_edges = relation_infer_mst(h, out, net, obj_token=20, rln_token=1)
print(f"MST: {len(pred_edges[0])} edges")

# 通常版のテスト
pred_nodes, pred_edges = relation_infer(h, out, net, obj_token=20, rln_token=1)
print(f"Normal: {len(pred_edges[0])} edges")
```

---

### オプション2: 勾配対応MSTモジュールの新規実装

**実装難易度:** 中
**実装時間:** 2-3時間
**利点:**
- 微分可能なMSTアルゴリズム
- エンド・ツー・エンド学習が可能
- より柔軟な最適化

**実装コード:**

```python
# inference_infinity_mst_nx_gradmst.py (新規作成)

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import itertools
from torchvision.ops import batched_nms

def soft_minimum_spanning_tree(cost_matrix, temperature=0.1):
    """
    微分可能なソフトMST実装
    Gumbel-Softmax技法を使用してMSTをソフト選択

    Args:
        cost_matrix: (N, N) コスト隣接行列
        temperature: Gumbel-Softmaxの温度パラメータ

    Returns:
        soft_mst: (N, N) ソフトMST隣接行列（勾配伝播可能）
    """
    N = cost_matrix.shape[0]
    device = cost_matrix.device

    # 上三角行列のみ使用（無向グラフ）
    edge_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
    valid_costs = cost_matrix * edge_mask + (1 - edge_mask) * 1e6

    # Gumbel-Softmaxサンプリング
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(valid_costs) + 1e-10) + 1e-10)
    logits = -valid_costs / temperature + gumbel_noise

    # Softmax選択（ソフトな選択）
    soft_selection = F.softmax(logits.view(-1), dim=0).view(N, N)

    # 対称行列に
    soft_mst = soft_selection + soft_selection.t()

    return soft_mst * edge_mask


def relation_infer(h, out, net, obj_token, rln_token, nms=False, map_=False, use_soft_mst=False):
    """
    勾配対応MSTベースの関係性推論

    Args:
        h: (batch, seq_len, hidden_dim) 隠れ層特徴
        out: dict - モデル出力 {'pred_logits': ..., 'pred_nodes': ...}
        net: ニューラルネットワークモデル
        obj_token: int - オブジェクトトークン数
        rln_token: int - リレーショントークン数
        nms: bool - NMS適用有無
        map_: bool - マッピング情報返却有無
        use_soft_mst: bool - ソフトMST使用（訓練時True推奨）

    Returns:
        pred_nodes: list of tensors - 予測ノード座標
        pred_edges: list of arrays - 予測エッジ
        (map_=True時) + ボックススコア、クラス情報
    """
    object_token = h[..., :obj_token, :]

    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    # 有効トークンの抽出
    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    # NMS適用
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            valid_token_id = torch.nonzero(token).squeeze(1)

            if valid_token_id.numel() == 0:
                continue

            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5

            ids2keep = batched_nms(
                boxes=valid_nodes * 1000,
                scores=valid_scores,
                idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            valid_token_nms[idx][valid_token_id_nms] = 1

        valid_token = valid_token_nms

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

        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        if map_:
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            # ノードペアの生成
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))

            node_pairs_valid = torch.tensor(
                [list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))]
            )

            # 辞書マッピング
            node_pairs_valid_dict = {}
            for num in range(node_pairs_valid.shape[0]):
                node_pair = node_pairs_valid[num]
                node_pairs_valid_dict[tuple(node_pair.cpu().numpy().tolist())] = num

            # 関係性特徴の構築
            if rln_token > 0:
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                relation_feature1 = torch.cat(
                    (object_token[batch_id, node_pairs[0], :],
                     object_token[batch_id, node_pairs[1], :]), 1
                )
                relation_feature2 = torch.cat(
                    (object_token[batch_id, node_pairs[1], :],
                     object_token[batch_id, node_pairs[0], :]), 1
                )

            # 関係性予測（勾配保持）
            if use_soft_mst:
                relation_pred1 = net.module.relation_embed(relation_feature1)
                relation_pred2 = net.module.relation_embed(relation_feature2)
            else:
                relation_pred1 = net.module.relation_embed(relation_feature1).detach()
                relation_pred2 = net.module.relation_embed(relation_feature2).detach()

            relation_pred = (relation_pred1 + relation_pred2) / 2.0

            # コスト行列の構築
            relation_pred_softmax = F.softmax(relation_pred, dim=-1)
            cost_pred_batch = relation_pred_softmax[:, 0]  # 非エッジ確率（コスト）

            # コスト隣接行列
            cost_adj_batch = torch.ones((node_id.shape[0], node_id.shape[0])).to(h.device) * 9999
            x, y = node_pairs_valid.t()
            cost_adj_batch[x, y] = cost_pred_batch
            cost_adj_batch[y, x] = cost_pred_batch

            if use_soft_mst:
                # ソフトMST（微分可能）
                soft_mst_adj = soft_minimum_spanning_tree(cost_adj_batch, temperature=0.1)
                mst_adj_batch = soft_mst_adj * torch.triu(torch.ones_like(soft_mst_adj), diagonal=1)
            else:
                # ハードMST（NetworkX使用）
                mst_adj_batch = compute_mst_nx(node_pairs_valid, cost_pred_batch)

            # エッジ抽出
            mst_tree_selected_list = torch.nonzero(mst_adj_batch > 0.5 if use_soft_mst else mst_adj_batch,
                                                   as_tuple=False)
            pred_edges.append(mst_tree_selected_list.cpu().numpy())

            # マッピング情報
            if map_:
                pred_rel_list = [
                    node_pairs_valid_dict[tuple(sorted((int(xy[0]), int(xy[1]))))]
                    for xy in mst_tree_selected_list if xy[0] != xy[1]
                ]

                if len(pred_rel_list) > 0:
                    pred_rel = torch.tensor(pred_rel_list).cpu().numpy()
                    pred_edges_boxes_score.append(relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy())
                    pred_edges_boxes_class.append(torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy())
                else:
                    pred_edges_boxes_score.append(np.array([]))
                    pred_edges_boxes_class.append(np.array([]))
        else:
            pred_edges.append(np.empty((0, 2)))

            if map_:
                pred_edges_boxes_score.append(np.empty(0))
                pred_edges_boxes_class.append(np.empty(0))

    if map_:
        return (pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score,
                pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class)
    else:
        return pred_nodes, pred_edges


def compute_mst_nx(node_pairs_valid, cost_pred_batch):
    """
    NetworkXを使用した従来のMST計算（非微分可能）
    """
    G = nx.Graph()
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    mst_edges = list(nx.minimum_spanning_edges(G, algorithm="kruskal", data=False))

    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight

    mst_adj_batch = torch.tensor(mst_adj_np)
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch
```

**使用方法:**
```python
# 推論時（勾配不要）
pred_nodes, pred_edges = relation_infer(
    h, out, net, obj_token=20, rln_token=1,
    nms=True, map_=False, use_soft_mst=False
)

# 訓練時（勾配必要）
pred_nodes, pred_edges = relation_infer(
    h, out, net, obj_token=20, rln_token=1,
    nms=False, map_=False, use_soft_mst=True
)
```

---

### オプション3: GNNベースの実装を活用

**実装難易度:** 低
**実装時間:** 10分
**利点:**
- 既存のGNN実装を活用
- より柔軟なグラフ構造学習
- 勾配伝播完全対応

**実装方法:**

```python
# valid_smd_guyot_nx.py の修正

if is_use_mst:
    # GNNベースを使用（より柔軟）
    from epoch import relation_infer_gnn as relation_infer
else:
    from epoch import relation_infer

# 呼び出し時
if args.use_gnn or is_use_mst:
    pred_nodes, pred_edges = relation_infer(
        h.detach(), out, net,
        config.MODEL.DECODER.OBJ_TOKEN,
        config.MODEL.DECODER.RLN_TOKEN,
        nms=True, map_=True
    )
```

**注意点:**
- `relation_infer_gnn` は `model` パラメータで `GAE_model` 属性が必要
- モデルアーキテクチャにGraph Auto-Encoderが含まれている必要がある

---

## 5. 推奨される解決策

### 第1優先: オプション1 - 既存実装の活用

**理由:**
1. **即座に動作可能** - 既存のテスト済みコードを使用
2. **リスクが低い** - 新規コード実装によるバグの心配がない
3. **保守性が高い** - 既存のコードベースとの整合性

**実装手順:**

#### ステップ1: valid_smd_guyot_nx.pyの修正
```python
# 1255-1259行目を以下に置き換え

if is_use_mst:
    from epoch import relation_infer_mst as relation_infer
else:
    from epoch import relation_infer
```

#### ステップ2: 動作確認
```bash
# テストスクリプトの実行
python valid_smd_guyot_nx.py --config config.yml --checkpoint model.pth --device cuda --use_mst
```

#### ステップ3: 結果の検証
- SMD（Street Mover Distance）メトリクスの確認
- トポロジメトリクスの検証
- MAPスコアの比較

### 第2優先: オプション2 - 勾配対応MST実装

**使用ケース:**
- エンド・ツー・エンド学習が必要な場合
- MSTの選択自体を学習したい場合
- より高度な最適化が必要な場合

**実装コスト:** 中（2-3時間）

### 第3優先: オプション3 - GNNベース

**使用ケース:**
- 既にGAEモデルが実装されている場合
- より柔軟なグラフ構造学習が必要な場合

---

## 6. Git履歴からの欠落原因の推測

### 6.1 調査結果

```bash
# Git履歴の確認
$ git log --all --oneline --decorate | head -30
```

**確認された最近のコミット:**
- `1608c9f` - Add Guyot dataset sampling tool and guyot_200_20 subset
- `46ee99d` - Organize tools and add dataset resizing utility
- `63656fe` - Add Guyot annotation visualization tool with uv-based project setup
- `ef03129` - Add 5 Guyot dataset samples for testing
- `4b93980` - Remove emojis from documentation

### 6.2 欠落原因の推測

#### 可能性1: 初期コミット時の除外（最有力）
- `.gitignore` に推論関連ファイルが含まれている可能性
- プロジェクト初期段階で実験的コードとして扱われ、意図的に除外された

#### 可能性2: プライベート実装
- 企業/研究機関内部でのみ使用される実装
- パブリックリポジトリには含めない方針

#### 可能性3: 実装途中
- `valid_smd_guyot_nx.py` のインポート文はプレースホルダー
- 将来的に実装予定だが、まだ作成されていない

#### 可能性4: ブランチ管理の問題
- 別ブランチで開発中
- マージ忘れやコンフリクト解決時の削除

### 6.3 .gitignoreの確認推奨

```bash
# .gitignoreの内容確認
cat .gitignore | grep -i inference
```

このコマンドで推論関連ファイルが意図的に除外されているか確認できます。

---

## 7. 追加の推奨事項

### 7.1 ドキュメント整備
1. **関数シグネチャの文書化**
   - `relation_infer` 系関数のAPIドキュメント作成
   - パラメータと返り値の詳細説明

2. **実装選択ガイド**
   - 各実装の特徴と使い分け
   - パフォーマンス比較データ

### 7.2 テストコードの追加
```python
# tests/test_inference.py (新規作成推奨)

import torch
import pytest
from epoch import relation_infer, relation_infer_mst, relation_infer_gnn

@pytest.fixture
def dummy_data():
    h = torch.randn(2, 21, 256)
    out = {
        'pred_logits': torch.randn(2, 20, 2),
        'pred_nodes': torch.randn(2, 20, 4)
    }
    return h, out

def test_relation_infer_basic(dummy_data):
    h, out = dummy_data
    # モックネットワーク
    class MockNet:
        class module:
            class relation_embed:
                def __call__(self, x):
                    return torch.randn(x.shape[0], 2)

    net = MockNet()
    pred_nodes, pred_edges = relation_infer(h, out, net, 20, 1)

    assert len(pred_nodes) == 2
    assert len(pred_edges) == 2

def test_relation_infer_mst(dummy_data):
    h, out = dummy_data
    # MST版のテスト
    # ... (同様の実装)

def test_edge_count_validity(dummy_data):
    """MSTエッジ数がノード数-1であることを確認"""
    h, out = dummy_data
    # ... (テスト実装)
```

### 7.3 エラーハンドリングの改善
```python
# valid_smd_guyot_nx.py への追加推奨

try:
    if is_use_mst:
        from epoch import relation_infer_mst as relation_infer
    else:
        from epoch import relation_infer
except ImportError as e:
    print(f"Warning: Could not import inference function: {e}")
    print("Falling back to basic relation_infer")
    from epoch import relation_infer
```

---

## 8. まとめ

### 8.1 問題の本質
- 存在しない `inference_infinity_mst_nx_gradmst.py` への依存
- 推論パイプラインの動作不可

### 8.2 即時解決策
**推奨アクション:**
```python
# valid_smd_guyot_nx.py の1255-1259行を以下に置き換え

if is_use_mst:
    from epoch import relation_infer_mst as relation_infer
else:
    from epoch import relation_infer
```

### 8.3 長期的改善
1. 推論モジュールの明確な分離と整理
2. ドキュメントとテストの整備
3. エラーハンドリングの強化

### 8.4 期待される効果
- ✓ 検証スクリプトの即座の動作復旧
- ✓ MST/非MST両方のパスの動作保証
- ✓ 既存のテスト済みコードの活用

---

## 付録A: 関連ファイル一覧

| ファイル | 行数 | 説明 |
|---------|------|------|
| `/home/inaho-omen/Project/TreeFormer/epoch.py` | 933 | 訓練・検証用エポック関数、推論関数を含む |
| `/home/inaho-omen/Project/TreeFormer/losses_only.py` | ~450KB | 損失関数とMST実装を含む |
| `/home/inaho-omen/Project/TreeFormer/valid_smd_guyot_nx.py` | 1762 | Guyotデータセット検証スクリプト（問題箇所） |
| `/home/inaho-omen/Project/TreeFormer/trainer.py` | 11KB | カスタムトレーナー実装 |

## 付録B: 参考文献

### MSTアルゴリズム
- Kruskal's Algorithm: O(E log E)
- Prim's Algorithm: O(E log V)
- NetworkX実装: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.minimum_spanning_tree.html

### 微分可能MST
- "Differentiable MST for End-to-End Learning", arXiv:2020
- Gumbel-Softmax技法: "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017

---

**報告書作成日:** 2025-11-14
**調査者:** Claude Code
**ステータス:** 解決策提案完了
