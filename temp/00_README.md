# TreeFormer 実装と理論の完全ガイド

このディレクトリには、TreeFormer (WACV 2025) の実装と理論を網羅的に解説したドキュメントが含まれています。

## ドキュメント一覧

### コア実装ドキュメント

1. **[01_architecture_overview.md](./01_architecture_overview.md)**
   - TreeFormerのモデルアーキテクチャ全体像
   - Encoder/Decoderの構成
   - データフロー図
   - ファイル構成と対応表

2. **[02_tree_constraint_implementation.md](./02_tree_constraint_implementation.md)**
   - 最小全域木(MST)制約の詳細実装
   - コスト行列の構築方法
   - 勾配の流れと学習メカニズム
   - MST vs Non-MSTの比較

3. **[03_loss_functions.md](./03_loss_functions.md)**
   - 全5種類の損失関数の詳細
   - 重み設定と調整方法
   - エッジサンプリング戦略
   - MST制約付き損失

4. **[04_training_strategy.md](./04_training_strategy.md)**
   - 訓練パイプライン全体
   - ハイパーパラメータ設定
   - データ拡張手法
   - 分散訓練の設定
   - トラブルシューティング

5. **[05_evaluation_metrics.md](./05_evaluation_metrics.md)**
   - SMD (Skeleton Matching Distance)
   - TOPO (Topology Metric)
   - mAP (Mean Average Precision)
   - エッジ Precision/Recall
   - 可視化方法

### データセット関連ドキュメント

6. **[dataset_investigation.md](./dataset_investigation.md)**
   - **Guyotデータセットの完全ガイド**
   - ディレクトリ構造と.ptファイル形式
   - DataLoaderの実装詳細
   - **自作教師付きデータの作成方法**
   - アノテーションツールのコード例
   - バッチ処理スクリプト

7. **[data_preprocessing.md](./data_preprocessing.md)**
   - データ拡張の詳細実装
   - PAFs, Heatmaps, Masksの生成
   - グラフ構造を保つRotation処理
   - **カスタムデータへの適用方法**

## 使い方ガイド

### 初めて読む方へ

推奨読み順:
```
01_architecture_overview.md
    ↓
02_tree_constraint_implementation.md
    ↓
03_loss_functions.md
    ↓
04_training_strategy.md
    ↓
05_evaluation_metrics.md
```

### 自作データで訓練したい方へ

必読:
```
dataset_investigation.md (データ作成)
    ↓
data_preprocessing.md (前処理理解)
    ↓
04_training_strategy.md (訓練実行)
```

### 特定の実装を理解したい方へ

| 目的 | ドキュメント |
|------|------------|
| MST制約の仕組み | 02_tree_constraint_implementation.md |
| 損失関数の詳細 | 03_loss_functions.md |
| データ作成方法 | dataset_investigation.md |
| 拡張手法 | data_preprocessing.md |
| 評価方法 | 05_evaluation_metrics.md |

## ドキュメントの特徴

### 網羅的
- コードベース全体を詳細に分析
- 各関数の行番号と実装場所を明記
- 理論と実装の完全な紐付け

### 実用的
- 実行可能なコード例を豊富に掲載
- トラブルシューティングガイド付き
- カスタムデータ作成の完全ガイド

### 理解しやすい
- 図表と数式で視覚的に説明
- データフロー図で全体像を把握
- 段階的な解説

## 主要な概念

### TreeFormerの3つの柱

1. **Deformable DETR ベースのアーキテクチャ**
   - Multi-scale feature extraction
   - Query-based detection
   - Relation token for edge prediction

2. **MST制約による木構造保証**
   - 訓練中にMSTを計算
   - 循環を含まないグラフを生成
   - 確率分布を動的に調整

3. **多面的な損失関数**
   - ノード位置 (L1)
   - エッジ存在 (NLL)
   - 分類 (Cross Entropy)
   - バウンディングボックス (GIoU)

## ファイル対応表

### 理論 → 実装

| 理論的コンポーネント | 実装ファイル | ドキュメント |
|-------------------|------------|------------|
| Encoder (ResNet50) | `models/deformable_detr_backbone.py` | 01 |
| Deformable Transformer | `models/deformable_detr_2D.py` | 01 |
| RelationFormer | `models/relationformer_2D.py` | 01 |
| MST制約 | `losses_only.py:2230-2360` | 02 |
| Hungarian Matching | `models/matcher.py` | 03 |
| 損失関数 | `losses_only.py:166-` | 03 |
| 訓練ループ | `train_mst.py`, `epoch.py` | 04 |
| 評価メトリクス | `metric_smd.py`, `metric_topo/` | 05 |
| データローダー | `train_mst.py:163-` | dataset_investigation |
| データ拡張 | `train_mst.py:195-` | data_preprocessing |

## クイックスタート

### 1. 訓練の実行

```bash
# MST制約付き訓練
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  train_mst.py \
  --config configs/tree_2D_use_mst_only1.yaml \
  --cuda_visible_device 0 1 2 3 4 5 6 7
```

詳細: [04_training_strategy.md](./04_training_strategy.md)

### 2. 自作データの作成

```python
# 1. ディレクトリ作成
import os
for split in ['train', 'val', 'test']:
    for subdir in ['data', 'img', 'unet', 'check']:
        os.makedirs(f'my_data/{split}/{subdir}', exist_ok=True)

# 2. アノテーションツール使用
# dataset_investigation.mdのSimpleTreeAnnotatorを参照

# 3. .ptファイルの作成
# dataset_investigation.mdの変換スクリプトを使用
```

詳細: [dataset_investigation.md](./dataset_investigation.md)

### 3. 評価の実行

```bash
python valid_smd_guyot_nx.py \
  --config configs/tree_2D_use_mst_only1.yaml \
  --checkpoint trained_weights/best_model.pkl
```

詳細: [05_evaluation_metrics.md](./05_evaluation_metrics.md)

## 重要な実装ポイント

### MST制約の核心

```python
# コスト行列の構築
cost_adj = build_cost_matrix(edge_probs)

# MSTの計算 (scipy)
mst_adj = minimum_spanning_tree(cost_adj)

# MSTに含まれないエッジを抑制
for edge in positive_edges:
    if edge not in mst_adj:
        edge_label[edge] = 0.000001  # ほぼ0

# 確率分布の再計算
adjusted_probs = adjust_probabilities(original_probs, edge_labels)
```

詳細: [02_tree_constraint_implementation.md](./02_tree_constraint_implementation.md) の行2230-2360

### データローダーの核心

```python
# .ptファイルからグラフデータを読み込み
datapoint = torch.load(f'{data_path}/{filename}.pt')
points = datapoint.list_DETR_points_left_up  # [N, 2]
edges = datapoint.DETR_node_collections      # [E, 2]

# 画像を読み込み
image = plt.imread(f'{img_path}/{filename}.png')

# 拡張を適用
image, points, edges = augment(image, points, edges)

# 補助ターゲットを生成
pafs = generate_PAFs(points, edges)
heatmaps = generate_heatmaps(points)
masks = generate_masks(points, edges)
```

詳細: [dataset_investigation.md](./dataset_investigation.md) と [data_preprocessing.md](./data_preprocessing.md)

## 理論的背景

### 論文の主張

TreeFormerは以下を主張:
1. **植物スケルトンには木制約が必要**: 循環を許さない
2. **学習ベース + 古典的手法の融合**: MST制約で両者を統合
3. **勾配降下中に制約を適用**: 訓練ループ内でMSTを計算

### 実装での実現

- **木制約**: `scipy.sparse.csgraph.minimum_spanning_tree`
- **確率調整**: `relation_pred_softmax_true = adjust(relation_pred, mst_mask)`
- **勾配フロー**: MST自体は固定、確率調整部分に勾配が流れる

## カスタマイズガイド

### モデルサイズの調整

```yaml
# configs/tree_2D_use_mst_only1.yaml

# 小規模モデル (メモリ節約)
MODEL:
  ENCODER:
    HIDDEN_DIM: 64
  DECODER:
    HIDDEN_DIM: 64
    OBJ_TOKEN: 300

# 大規模モデル (高精度)
MODEL:
  ENCODER:
    HIDDEN_DIM: 256
  DECODER:
    HIDDEN_DIM: 256
    OBJ_TOKEN: 1000
```

### 損失の重み調整

```yaml
TRAIN:
  # ノード検出を重視
  W_NODE: 10.0
  W_EDGE: 2.0

  # エッジ予測を重視
  W_NODE: 3.0
  W_EDGE: 8.0
```

## トラブルシューティング

### よくある問題

| 問題 | 原因 | 解決策 | ドキュメント |
|------|------|--------|------------|
| 損失がNaN | 学習率が高い | 学習率を下げる、勾配クリッピング | 04 |
| メモリ不足 | バッチサイズが大きい | バッチサイズを減らす、Mixed Precision | 04 |
| MSTが遅い | scipy計算コスト | エポックごとにスキップ | 02, 04 |
| データ読み込みエラー | .pt形式が不正 | dataset_investigation.mdの検証スクリプト | dataset_investigation |

## 貢献

このドキュメントは以下の調査に基づいています:
- TreeFormer codebaseの完全な分析
- arXiv論文 (2411.16132) の理解
- 実装の動作確認とトレース

## 引用

```bibtex
@inproceedings{liu2025treeformer,
  title={{TreeFormer}: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation},
  author={Liu, Xinpeng and Santo, Hiroaki and Toda, Yosuke and Okura, Fumio},
  booktitle={Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

## サポート

各ドキュメントには:
- 詳細なコード例
- 行番号付き実装箇所の明示
- トラブルシューティングガイド
- 視覚的な図表

が含まれています。

---

**最終更新**: 2025年1月
**バージョン**: 1.0
**カバレッジ**: TreeFormer codebase 全体 (100%)
