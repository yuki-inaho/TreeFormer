# TreeFormer: アーキテクチャ概要

## 論文情報
- **タイトル**: TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation
- **会議**: WACV 2025
- **著者**: Xinpeng Liu, Hiroaki Santo, Yosuke Toda, Fumio Okura (Osaka University, Phytometrics, Nagoya University)
- **arXiv**: https://arxiv.org/abs/2411.16132

## コアコンセプト

TreeFormerは、単一画像から植物のスケルトン(骨格)構造を推定する手法です。重要な特徴は、**学習ベースのグラフ生成と従来のグラフアルゴリズムを組み合わせた木制約**です。

### 主要な革新点

1. **木制約グラフ生成**
   - 訓練ループ中に、生成されたグラフを最小全域木(MST)に射影
   - 勾配降下の過程に直接グラフ理論の制約を統合
   - 望ましくない特徴値を抑制することで、木構造を保証

2. **なぜ木制約が必要か**
   - 植物構造は人間の骨格のような固定トポロジーを持たない
   - 木構造(ルートから分岐)という制約が自然
   - 従来の方法では循環(サイクル)を含むグラフが生成されることがある

## モデルアーキテクチャ

TreeFormerは以下のコンポーネントで構成されます:

### 1. バックボーン (Encoder)
- **ファイル**: `models/deformable_detr_backbone.py`
- **タイプ**: ResNet50ベースのDeformable Transformer Backbone
- **機能**:
  - 入力画像からマルチスケール特徴を抽出
  - 4つの特徴レベル(8x8, 4x4, 2x2, 1x1)を生成
- **実装**: `build_backbone()`関数

```python
# 設定例 (configs/tree_2D_use_mst_only1.yaml)
ENCODER:
  TYPE: deformable_transformer_backbone
  HIDDEN_DIM: 128
  BACKBONE: resnet50
  NUM_FEATURE_LEVELS: 4
```

### 2. Deformable Transformer (Decoder)
- **ファイル**: `models/deformable_detr_2D.py`
- **主要クラス**: `DeformableTransformer`
- **コンポーネント**:
  - **Encoder**: Multi-scale Deformable Attention でマルチスケール特徴を処理
  - **Decoder**: クエリトークンを使ってノードとエッジを予測

```python
# Decoderの構成
DECODER:
  HIDDEN_DIM: 128
  NHEADS: 8
  ENC_LAYERS: 4
  DEC_LAYERS: 4
  DIM_FEEDFORWARD: 128
  OBJ_TOKEN: 600    # 最大ノード数
  RLN_TOKEN: 1      # Relation token
  RLN_ATTN: True    # Relation attention有効化
```

### 3. RelationFormer
- **ファイル**: `models/relationformer_2D.py`
- **主要クラス**: `RelationFormer`
- **機能**:
  - Object tokens: ノード検出用
  - Relation token: エッジ予測用
  - 各トークンから分類・位置・関係性を予測

**出力ヘッド**:
- `class_embed`: ノードの有無を分類 (2クラス)
- `bbox_embed`: ノードの位置を予測 (x, y, w, h)
- `relation_embed`: エッジの有無を予測 (2クラス)

### 4. Hungarian Matcher
- **ファイル**: `models/matcher.py`
- **クラス**: `HungarianMatcher`
- **目的**: 予測とGround Truthの最適マッチングを計算
- **コスト関数**:
  ```python
  C = cost_nodes * weight_nodes + cost_class * weight_class
  ```
  - `cost_nodes`: L1距離でノード位置の差を計算
  - `cost_class`: 分類のクロスエントロピー

## データフロー

```
入力画像 (512x512)
    ↓
Backbone (ResNet50)
    ↓
Multi-scale features (4 levels)
    ↓
Deformable Transformer Encoder
    ↓
Deformable Transformer Decoder
    ↓
Object Tokens (600個) + Relation Token (1個)
    ↓
┌─────────────┬──────────────┐
↓             ↓              ↓
Class        Nodes          Edges
(2クラス)    (x,y,w,h)      (2クラス)
    ↓             ↓              ↓
Hungarian Matching (予測とGTのマッチング)
    ↓
Loss計算 + MST制約の適用
```

## ファイル構成

```
TreeFormer/
├── models/
│   ├── relationformer_2D.py       # メインモデル
│   ├── deformable_detr_2D.py      # Transformer実装
│   ├── deformable_detr_backbone.py # Backbone
│   ├── matcher.py                  # Hungarian Matcher
│   ├── position_encoding_2D.py     # 位置エンコーディング
│   └── ops/                        # Deformable Attention
│       └── modules/ms_deform_attn.py
├── losses_only.py                  # 損失関数(MST制約含む)
├── train_mst.py                    # MST制約付き訓練
├── train_unmst.py                  # 通常訓練
├── valid_smd_guyot_nx.py          # 評価スクリプト
├── metric_topo/                    # トポロジー評価
│   ├── topo.py                     # TOPO metric
│   └── graph.py                    # グラフ操作
└── configs/
    ├── tree_2D_use_mst_only1.yaml    # MST制約設定
    └── tree_2D_use_unmst_only1.yaml  # 通常設定
```

## 次のドキュメント

- [02_tree_constraint_implementation.md](./02_tree_constraint_implementation.md): 木制約の実装詳細
- [03_loss_functions.md](./03_loss_functions.md): 損失関数の詳細
- [04_training_strategy.md](./04_training_strategy.md): 訓練戦略
- [05_evaluation_metrics.md](./05_evaluation_metrics.md): 評価指標
