# 木制約の実装詳細

## 概要

TreeFormerの核心は**最小全域木(MST)を用いた木制約**です。この制約により、モデルは訓練中に循環のない木構造のグラフのみを生成するよう学習します。

## 理論的背景

### なぜMSTか?

植物のスケルトンは以下の特性を持ちます:
1. **根が存在**: 地面から伸びる主幹
2. **分岐構造**: 枝が分かれていく
3. **循環なし**: 枝が再び結合することはない

これらの特性は、グラフ理論における**木(Tree)**の定義と一致します:
- 連結グラフ
- 循環(サイクル)なし
- N個のノードに対してN-1個のエッジ

### MSTによる制約

**最小全域木(Minimum Spanning Tree, MST)**は:
- グラフの全ノードを連結
- エッジの重みの合計が最小
- 循環を含まない

モデルが予測したエッジ確率をコストとしてMSTを計算することで、**最も確からしい木構造**を取り出せます。

## 実装の詳細

### 1. コスト隣接行列の構築

**場所**: `losses_only.py:2230-2236`

```python
# 予測されたエッジ確率からコスト行列を構築
cost_adj_batch = torch.zeros((num_nodes, num_nodes))
relation_pred_softmax_batch = F.softmax(relation_pred_batch, dim=-1)
cost_pred_batch = relation_pred_softmax_batch[:, 0]  # クラス0 = エッジなし = コスト

for num_pairs in range(all_edges_.shape[0]):
    x, y = all_edges_[num_pairs]
    cost_adj_batch[x, y] = cost_pred_batch[num_pairs]
    cost_adj_batch[y, x] = cost_pred_batch[num_pairs]

# 上三角行列のみを使用(無向グラフ)
cost_adj_batch = cost_adj_batch * torch.triu(torch.ones_like(cost_adj_batch), diagonal=1)
```

**重要**:
- `relation_pred_softmax_batch[:, 0]`: "エッジなし"の確率 → コストとして使用
- コストが低い = エッジが存在する確率が高い
- 対称行列として構築(無向グラフ)

### 2. MSTの計算

**場所**: `losses_only.py:2241-2242`

```python
from scipy.sparse.csgraph import minimum_spanning_tree

# scipyのMSTアルゴリズムを使用
mst_adj_batch = minimum_spanning_tree(cost_adj_batch.cpu().numpy().copy())
mst_adj_batch = (mst_adj_batch + mst_adj_batch.T).toarray()
```

**アルゴリズム**: scipyは内部でKruskalアルゴリズムを使用
- 時間計算量: O(E log V)
- Eはエッジ数、Vはノード数

**代替実装**: `prims_mst()`
- **場所**: `losses_only.py:76-100`
- PyTorchで実装されたPrimのアルゴリズム
- ただし、実際にはscipyの方が高速なため未使用

### 3. エッジラベルの調整

**場所**: `losses_only.py:2246-2253`

```python
# MSTに含まれないエッジのラベルを抑制
mst_edge_label_batch = torch.ones(pos_edge.shape[0])

for pos_pairs in range(pos_edge.shape[0]):
    x, y = pos_edge[pos_pairs]
    if mst_adj_batch[x, y] == 0:  # MSTに含まれない
        mst_edge_label_batch[pos_pairs] = 0.000001  # ほぼ0に設定
```

**仕組み**:
- Ground Truthのエッジのうち、MSTに含まれないものを特定
- それらのエッジの目標ラベルを0.000001に設定
- これにより、MSTに含まれないエッジは学習されない

### 4. 確率分布の再計算

**場所**: `losses_only.py:2256-2266`

```python
# エッジ確率を調整
relation_pred_softmax_batch_true = F.softmax(relation_pred_batch, dim=-1).clone()

# MSTに含まれないエッジの"エッジあり"確率を抑制
relation_pred_softmax_batch_true[:, 1] = \
    relation_pred_softmax_batch_true[:, 1] * mst_edge_label_batch

# 抑制された確率を"エッジなし"に再配分
relation_pred_softmax_batch_true[:, 0] = \
    relation_pred_softmax_batch_true[:, 0] + \
    relation_pred_softmax_batch_true[:, 1] * (1 - mst_edge_label_batch)
```

**確率の再配分**:
```
元の予測: [P(エッジなし), P(エッジあり)]
         ↓ MSTに含まれない場合
調整後:   [P(エッジなし) + P(エッジあり) * 0.999999, P(エッジあり) * 0.000001]
```

これにより、確率の合計が1に保たれます。

### 5. 損失の計算

**場所**: `losses_only.py:2273-2275`

```python
# NLL Loss (Negative Log Likelihood)
nllloss_func = nn.NLLLoss(reduction='mean')
relation_pred_log_softmax_batch_true = relation_pred_softmax_batch_true.log()
nlloss_batch = nllloss_func(relation_pred_log_softmax_batch_true, edge_label_batch)
```

## 勾配の流れ

重要な点: **MST計算自体は微分不可能**ですが、勾配は以下のように流れます:

```
Forward:
  予測 → Softmax → コスト行列 → MST計算 → ラベル調整 → 確率再計算 → Loss

Backward:
  Loss ← 確率再計算 ← ラベル調整 ← (MST計算はスキップ) ← Softmax ← 予測
          ↑
       ここから勾配が流れる
```

**メカニズム**:
1. MSTの計算には`.detach()`が適用されている
2. MST自体は固定されたマスクとして機能
3. モデルは「MSTに含まれるエッジの確率を上げる」ように学習
4. 結果として、次のイテレーションでは異なるMSTが選ばれる可能性がある

## 訓練時の動作

### Epoch 1-10: 初期段階
- ランダムな予測
- MSTは不安定に変化
- 徐々に正しいエッジの確率が上がる

### Epoch 10-50: 収束段階
- MSTが安定し始める
- 正しいエッジが一貫してMSTに含まれる
- 誤ったエッジは抑制される

### Epoch 50以降: 精緻化
- MSTがほぼ固定
- 細かな位置調整が行われる

## MST vs Non-MST訓練の比較

### MST制約あり (`train_mst.py`)
```yaml
# configs/tree_2D_use_mst_only1.yaml
--use_mst_train: True
```
- **利点**: 循環のない木構造を保証
- **欠点**: MSTの計算コストが追加

### MST制約なし (`train_unmst.py`)
```yaml
# configs/tree_2D_use_unmst_only1.yaml
--use_mst_train: False
```
- **利点**: 訓練が高速
- **欠点**: 循環を含む可能性がある

## コード内の関連箇所

| 機能 | ファイル | 行番号 | 説明 |
|------|---------|--------|------|
| MST計算(scipy) | `losses_only.py` | 10, 2241 | scipyのMST実装をインポート・使用 |
| MST計算(Prim) | `losses_only.py` | 76-100 | PyTorchでのPrim実装(未使用) |
| コスト行列構築 | `losses_only.py` | 2230-2236 | エッジ確率からコスト行列を作成 |
| ラベル調整 | `losses_only.py` | 2246-2253 | MSTベースでラベルを調整 |
| 確率再計算 | `losses_only.py` | 2256-2266 | 調整後の確率分布を計算 |
| Loss関数 | `losses_only.py` | 2362-2360 | `loss_edges_mst_new()` |
| 訓練スクリプト | `train_mst.py` | 787 | MST制約の適用フラグ |

## 視覚化

```
Ground Truth Edges:    Predicted Probs:     Cost Matrix:        MST:
   A---B---C              A-B: 0.9            A-B: 0.1          A---B---C
   |                      A-C: 0.7            A-C: 0.3                  |
   D                      B-C: 0.5            B-C: 0.5                  D
                          A-D: 0.8            A-D: 0.2
                          B-D: 0.3            B-D: 0.7
                          C-D: 0.6            C-D: 0.4
```

コストが最小の経路を選ぶことで、循環のない木構造を保証します。
