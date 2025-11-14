# TreeFormer データセット調査レポート

**日付:** 2025-11-14
**調査レベル:** 非常に詳細
**目的:** Guyotデータセット構造とデータローダー実装の包括的分析

---

## 目次

1. [データセットディレクトリ構造](#1-データセットディレクトリ構造)
2. [データファイル形式 (.ptファイル)](#2-データファイル形式-ptファイル)
3. [データローダー実装](#3-データローダー実装)
4. [データフローパイプライン](#4-データフローパイプライン)
5. [カスタム訓練データの作成](#5-カスタム訓練データの作成)
6. [コード例](#6-コード例)
7. [重要な実装の詳細](#7-重要な実装の詳細)

---

## 1. データセットディレクトリ構造

### 1.1 期待されるディレクトリレイアウト

Guyotデータセットは、train/val/test分割を持つ階層構造に従います：

```
guyot_data/
├── train/
│   ├── data/           # グラフアノテーションを含むPyTorchテンソルファイル (.pt)
│   │   ├── Set02_IMG_3468.pt
│   │   ├── Set02_IMG_3469.pt
│   │   └── ...
│   ├── img/            # 元の入力画像 (PNG形式)
│   │   ├── Set02_IMG_3468.png
│   │   ├── Set02_IMG_3469.png
│   │   └── ...
│   ├── unet/           # UNet予測 (補助的、メインローダーでは未使用)
│   │   └── images
│   └── check/          # 可視化/チェック画像 (補助的)
│       └── images
├── val/
│   ├── data/
│   ├── img/
│   ├── unet/
│   └── check/
└── test/
    ├── data/
    ├── img/
    ├── unet/
    └── check/
```

### 1.2 ファイル命名規則

- **画像ファイル**: `{name}.png` (例: `Set02_IMG_3468.png`)
- **データファイル**: `{name}.pt` (例: `Set02_IMG_3468.pt`)
- **対応関係**: 各`.pt`ファイルは、同じベース名を持つ対応する`.png`ファイルが必要

### 1.3 主要ディレクトリ

| ディレクトリ | 目的 | 形式 | 必須 |
|-----------|---------|--------|----------|
| `data/` | 正解グラフアノテーション | `.pt` PyTorchテンソル | はい |
| `img/` | 入力RGB/グレースケール画像 | `.png` | はい |
| `unet/` | UNetセグメンテーション出力 | `.png` | いいえ (レガシー) |
| `check/` | 可視化出力 | `.png` | いいえ (デバッグ) |

---

## 2. データファイル形式 (.ptファイル)

### 2.1 ファイル構造

各`.pt`ファイルには、以下の属性を持つ**Pythonオブジェクト**(辞書ではない)が含まれます：

```python
class DataPoint:
    list_DETR_points_left_up: torch.Tensor  # ノード座標 (正規化)
    DETR_node_collections: torch.Tensor      # エッジ接続
```

### 2.2 属性の詳細

#### 2.2.1 `list_DETR_points_left_up`

**型:** `torch.Tensor`
**形状:** `[N, 2]` (Nはノード数)
**データ型:** `torch.float32`
**座標系:** [0, 1]の範囲の正規化座標

- **列 0:** x座標 (画像幅で正規化)
- **列 1:** y座標 (画像高さで正規化)

**例:**
```python
tensor([[0.2350, 0.4120],  # ノード 0: x=0.235, y=0.412
        [0.3140, 0.5230],  # ノード 1
        [0.4560, 0.6780],  # ノード 2
        ...])
```

**ピクセル座標を取得するには:**
```python
pixel_coords = list_DETR_points_left_up * torch.tensor([width, height])
```

#### 2.2.2 `DETR_node_collections`

**型:** `torch.Tensor`
**形状:** `[E, 2]` (Eはエッジ数)
**データ型:** `torch.long`
**形式:** エッジリスト表現

- **列 0:** ソースノードのインデックス
- **列 1:** ターゲットノードのインデックス

**例:**
```python
tensor([[0, 1],   # ノード0からノード1へのエッジ
        [1, 2],   # ノード1からノード2へのエッジ
        [1, 3],   # ノード1からノード3へのエッジ (分岐)
        ...])
```

**制約:**
- **木構造**を形成 (連結、非巡回グラフ)
- ノード0は常に**ルートノード**
- エッジは木における親子関係を表す
- `nx.is_tree()`検証を通過する必要がある

### 2.3 .ptファイルの読み込み

```python
import torch

# データファイルを読み込む
datapoint = torch.load('path/to/data/Set02_IMG_3468.pt')

# 属性にアクセス
points = datapoint.list_DETR_points_left_up  # 形状: [N, 2]
edges = datapoint.DETR_node_collections       # 形状: [E, 2]

print(f"ノード数: {points.shape[0]}")
print(f"エッジ数: {edges.shape[0]}")
```

---

## 3. データローダー実装

### 3.1 データセットクラス: `LoadCNNDataset`

**場所:** `train_mst.py` (163-677行目) および `valid_smd_guyot_nx.py` (182-665行目)

#### 3.1.1 コンストラクタパラメータ

```python
class LoadCNNDataset(Dataset):
    def __init__(
        self,
        parent_path,              # train/val/testディレクトリへのパス
        max_size=1000,            # 最大画像サイズ
        max_change_light_rate=0.3, # 明るさ拡張の範囲
        is_train=True,            # 訓練用拡張を有効化
        is_rotate=False           # 回転拡張を有効化
    ):
```

#### 3.1.2 初期化プロセス

```python
self.parent_path = parent_path
self.tgt_data_path = os.path.join(parent_path, "data")  # .ptファイル
self.img_path = os.path.join(parent_path, "img")         # .pngファイル
self.file_list = self.processed_file_names               # .ptファイルのリスト

# 初期化時にすべてのグラフデータを読み込む
ids1, (list_DETR_points_left_up, list_DETR_node_collections) = \
    load_detr_dataset(self.tgt_data_path)

self.ids1 = ids1                                    # ファイル名
self.list_DETR_points_left_up = list_DETR_points_left_up  # すべてのノード座標
self.list_DETR_node_collections = list_DETR_node_collections  # すべてのエッジ
```

**重要な設計:** すべてのグラフアノテーションは、訓練中の高速アクセスのために初期化時にメモリに読み込まれます。

### 3.2 データ読み込み関数

**場所:** `train_mst.py` 144-160行目

```python
def load_detr_dataset(tgt_data_path):
    """
    dataディレクトリからすべての.ptファイルを読み込む。

    Args:
        tgt_data_path: .ptファイルを含む'data'ディレクトリへのパス

    Returns:
        ids: .ptファイル名のリスト
        (list_DETR_points_left_up, list_DETR_node_collections):
            すべてのサンプルのテンソルのリスト
    """
    path_list = []
    for file in os.listdir(tgt_data_path):
        path_list.append(file)

    list_DETR_points_left_up = []
    list_DETR_node_collections = []
    ids = path_list

    for id in ids:
        datapoint = torch.load(tgt_data_path + '/' + id)
        DETR_points_left_up = datapoint.list_DETR_points_left_up
        DETR_node_collections = datapoint.DETR_node_collections

        list_DETR_points_left_up.append(DETR_points_left_up)
        list_DETR_node_collections.append(DETR_node_collections)

    return ids, (list_DETR_points_left_up, list_DETR_node_collections)
```

### 3.3 `__getitem__` メソッド

**目的:** 単一サンプルを取得・処理する

**主要ステップ:**

1. **グラフデータの読み込み** (事前読み込みされたメモリから)
2. **画像の読み込み** (ディスクから)
3. **データ拡張** (訓練時)
4. **補助ターゲットの生成** (PAFs、ヒートマップ、マスク)
5. **画像の前処理** (正規化、リサイズ)

**コードフロー:**

```python
def __getitem__(self, idx):
    # 1. ファイル名を取得
    label_img_name = self.file_list[idx].split(".pt")[0] + ".png"
    label_img_name0 = label_img_name.split(".")[0]

    # 2. 事前読み込みされたグラフデータを取得
    list_DETR_points_left_up_idx = self.list_DETR_points_left_up[idx]
    list_DETR_node_collections_idx = self.list_DETR_node_collections[idx]

    # 3. 画像を読み込む
    plt_img = plt.imread(os.path.join(self.img_path, label_img_name))
    plt_img = plt_img.astype(np.float32)

    # RGBA画像を処理 (アルファチャンネルを削除)
    if len(plt_img.shape) == 3 and plt_img.shape[2] == 4:
        plt_img = plt_img[:, :, :3]

    height, width, channels = plt_img.shape

    # 4. データ拡張 (訓練時のみ)
    nodes_list = list_DETR_points_left_up_idx * torch.tensor([width, height])
    nodes_list = nodes_list.numpy()

    if self.is_train:
        result_list = self._augment_one_sample(input_img, nodes_list)
        feature_img, nodes = result_list[1], result_list[2]
        list_DETR_points_left_up = torch.tensor(nodes, dtype=torch.float)
    else:
        feature_img = input_img
        list_DETR_points_left_up = list_DETR_points_left_up_idx

    # 5. 画像の正規化
    if len(feature_img.shape) == 3 and feature_img.shape[2] == 3:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    feature_img = transform(feature_img)

    # 6. 画像のリサイズ
    C, height, width = feature_img.shape
    cut_height = height // 2
    cut_width = width // 2
    feature_img = TF.resize(feature_img, size=[cut_height, cut_width])

    # max_sizeを超える場合は更にリサイズ
    if max(cut_width, cut_height) > self.max_size:
        if cut_width > cut_height:
            scale = self.max_size / cut_width
            new_width = self.max_size
            new_height = int(cut_height * scale)
        else:
            scale = self.max_size / cut_height
            new_height = self.max_size
            new_width = int(cut_width * scale)
        feature_img = TF.resize(feature_img, size=[new_height, new_width])

    # 7. 補助ターゲットの生成 (PAFs、マスク、ヒートマップ)
    feature_size = (feature_img.shape[1], feature_img.shape[2])
    PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
        list_DETR_node_collections_idx=list_DETR_node_collections_idx,
        list_DETR_points_left_up_idx=list_DETR_points_left_up,
        feature_size=feature_size,
        sigma=3,
        unet_thickness=3,
        mask_thickness=6
    )

    # 8. すべてのコンポーネントを返す
    return (feature_img.contiguous(),
            label_img_name0,
            list_DETR_points_left_up,
            list_DETR_node_collections_idx,
            PAFs_idx, mask_idx, unet_idx, heatmap_idx,
            self.ids1[idx])
```

### 3.4 データ拡張メソッド

#### 3.4.1 明るさ調整

```python
def _changeLight(self, img):
    flag = random.uniform(
        1 - self.max_change_light_rate,
        1 + self.max_change_light_rate
    )
    return exposure.adjust_gamma(img, flag)
```

#### 3.4.2 ガウシアンノイズ

```python
def _gasuss_noise(self, image, mu=0.0, sigma=0.1):
    gasuss_img = image.astype(np.float32)
    noise = np.random.normal(mu, sigma, gasuss_img.shape)
    gauss_noise = gasuss_img + noise
    gauss_noise = np.clip(gauss_noise, 0.0, 1.0)
    return gauss_noise
```

#### 3.4.3 水平反転

```python
def _flip2(self, img, nodes_list):
    w = img.shape[1]
    img2 = cv2.flip(img, 1)  # 水平反転

    # ノード座標を調整
    flip_new_nodes_list = []
    for x, y in nodes_list:
        flip_new_nodes_list.append([w - x, y])

    return img2, flip_new_nodes_list
```

#### 3.4.4 回転 (オプション、複雑)

```python
def _rotate(self, img, nodes_tensor, connect_tensor):
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img2 = cv2.warpAffine(img, M, (width, height))

    # ノード座標を変換
    # 画像境界外のノードを削除
    # エッジ分岐拡張を追加
    # 木構造を検証

    return img2_tensor, final_nodes_tensor, final_connect_tensor, M
```

### 3.5 補助ターゲットの生成

#### 3.5.1 Part Affinity Fields (PAFs)

**目的:** エッジの方向と位置をエンコード

```python
def generate_PAFs(height, width, points, paths, line_thickness=2):
    PAFs = np.zeros((height, width, 2), dtype=np.float32)

    for branch in paths:
        for idx in range(len(branch) - 1):
            start_point = points[branch[idx]]
            end_point = points[branch[idx + 1]]

            # 単位ベクトルを計算
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            ux = (x2 - x1) / length
            uy = (y2 - y1) / length

            # ライン沿いに塗りつぶし
            for t in np.linspace(0, 1, int(length)):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < width and 0 <= y < height:
                    PAFs[y-thickness:y+thickness, x-thickness:x+thickness, 0] = ux
                    PAFs[y-thickness:y+thickness, x-thickness:x+thickness, 1] = uy

    return PAFs
```

**出力形状:** `[height, width, 2]`
**チャンネル 0:** エッジ方向のx成分
**チャンネル 1:** エッジ方向のy成分

#### 3.5.2 ヒートマップ

**目的:** ガウシアンカーネルでノード位置をエンコード

```python
def generate_heatmap(normalized_kpts, image_size, sigma):
    H, W = image_size
    heatmap = np.zeros((H, W))

    for keypoint in normalized_kpts:
        x_normalized, y_normalized = keypoint
        x = x_normalized * W
        y = y_normalized * H

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        gaussian = np.exp(-0.5 * ((xx - x)**2 + (yy - y)**2) / sigma**2)
        gaussian[gaussian < 0.01] = 0
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap
```

**出力形状:** `[height, width]`
**値:** ノード位置でのガウシアンピーク

#### 3.5.3 マスク

**目的:** スケルトン構造のバイナリマスク

```python
def create_mask_with_polylines(image_shape, keypoints, segments, thickness=2):
    kpts = keypoints.copy()
    kpts[:, 0] *= image_shape[1]  # ピクセル座標にスケール
    kpts[:, 1] *= image_shape[0]

    mask = np.zeros(image_shape, dtype=np.uint8)

    for segment in segments:
        segment_points = kpts[segment].reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask, [segment_points], isClosed=False,
                      color=1, thickness=thickness)

    return mask
```

**出力形状:** `[height, width]`
**値:** 0 (背景), 1 (スケルトン)

### 3.6 カスタムコレート関数

**場所:** `train_mst.py` 679-712行目

**目的:** 複数のサンプルを適切なフォーマットでバッチ化

```python
def custom_collate_fn(batch):
    (feature_img, label_img_name0, list_DETR_points_left_up,
     list_DETR_node_collections, list_PAFs, list_mask,
     list_unet, list_heatmap, ids1) = zip(*batch)

    ACT_1 = 0.9999999
    ACT_0 = 0.0000001

    # 画像: リストとして保持 (可変サイズ)
    images = [item.to(torch.float32) for item in feature_img]

    # グラフデータ: リストとして保持
    points_left_up = [item for item in list_DETR_points_left_up]
    edges = [item for item in list_DETR_node_collections]

    # 補助ターゲット: バッチテンソルに結合
    PAFs_list_transformed = [PAFs.unsqueeze(0).permute(0, 3, 1, 2)
                             for PAFs in list_PAFs]
    mask_list_transformed = [mask.unsqueeze(0).unsqueeze(0)
                             for mask in list_mask]
    unet_list_transformed = [unet.unsqueeze(0).unsqueeze(0)
                             for unet in list_unet]
    heatmap_list_transformed = [heatmap.unsqueeze(0).unsqueeze(0)
                                for heatmap in list_heatmap]

    PAFs_concatenated = torch.cat(PAFs_list_transformed, 0)
    mask_concatenated = torch.cat(mask_list_transformed, 0).contiguous()
    unet_concatenated = torch.cat(unet_list_transformed, 0)
    heatmap_concatenated = torch.cat(heatmap_list_transformed, 0)

    # 値をクランプ
    PAFs_concatenated = torch.clamp(PAFs_concatenated, min=-ACT_1, max=ACT_1)
    unet_concatenated = torch.clamp(unet_concatenated, min=ACT_0, max=ACT_1)
    heatmap_concatenated = torch.clamp(heatmap_concatenated, min=ACT_0, max=ACT_1)

    detr_ids = list(ids1)

    return [images, points_left_up, edges,
            PAFs_concatenated, mask_concatenated,
            unet_concatenated, heatmap_concatenated,
            detr_ids],
```

**戻り値形式:**
- `images`: テンソルのリスト (可変サイズ)
- `points_left_up`: テンソルのリスト [N, 2]
- `edges`: テンソルのリスト [E, 2]
- `PAFs_concatenated`: `[batch_size, 2, H, W]`
- `mask_concatenated`: `[batch_size, 1, H, W]`
- `unet_concatenated`: `[batch_size, 1, H, W]`
- `heatmap_concatenated`: `[batch_size, 1, H, W]`
- `detr_ids`: ファイル名のリスト

### 3.7 データローダーのインスタンス化

**場所:** `train_mst.py` 868-892行目

```python
# 訓練データセット
train_path = "/path/to/guyot_data/train"
dataset_train = LoadCNNDataset(
    parent_path=train_path,
    max_size=512,              # 設定から: DATA.MAX_SIZE
    max_change_light_rate=0.3,
    is_train=False,            # 拡張を有効にする場合はTrue
    is_rotate=True             # 回転拡張を有効化
)

# マルチGPU訓練用の分散サンプラー
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

# データローダー
train_loader = DataLoader(
    dataset_train,
    batch_size=8,              # 設定から: DATA.BATCH_SIZE
    shuffle=False,             # サンプラーがシャッフルを処理
    collate_fn=custom_collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
    sampler=train_sampler
)

# 検証データセット
val_path = "/path/to/guyot_data/val"
dataset_val = LoadCNNDataset(
    parent_path=val_path,
    max_size=512,
    max_change_light_rate=0.3,
    is_train=False,            # 検証では拡張なし
    is_rotate=False
)

valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)

val_loader = DataLoader(
    dataset_val,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
    sampler=valid_sampler
)
```

---

## 4. データフローパイプライン

### 4.1 訓練パイプライン

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. データセット初期化                                             │
├─────────────────────────────────────────────────────────────────┤
│  LoadCNNDataset(parent_path="guyot_data/train", ...)           │
│    ├─ data/ディレクトリで.ptファイルをスキャン                    │
│    ├─ すべてのグラフアノテーションをメモリに読み込み              │
│    └─ ファイルリストを保存                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. __getitem__(idx) - サンプルごと                               │
├─────────────────────────────────────────────────────────────────┤
│  a) メモリからグラフデータを取得                                  │
│     - points: [N, 2] 正規化座標                                  │
│     - edges: [E, 2] エッジリスト                                 │
│                                                                  │
│  b) ディスクから画像を読み込む                                    │
│     - {name}.pngを読む                                          │
│     - float32に変換、存在する場合はアルファを削除                 │
│                                                                  │
│  c) データ拡張 (is_train=Trueの場合)                             │
│     - ランダムな明るさ: ±30%ガンマ調整                           │
│     - ガウシアンノイズ: σ=0.1                                    │
│     - 水平反転: 50%の確率                                        │
│     - それに応じてノード座標を更新                                │
│                                                                  │
│  d) 画像の前処理                                                 │
│     - 正規化: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]        │
│     - リサイズ: height//2, width//2                             │
│     - max_size (512)を超える場合は更にリサイズ                   │
│                                                                  │
│  e) 補助ターゲットを生成                                         │
│     - セグメント抽出 (ルートからのDFS)                           │
│     - PAFs: エッジ方向フィールド [H, W, 2]                       │
│     - Heatmap: ガウシアンノード位置 [H, W]                       │
│     - Mask: バイナリスケルトン (thick=6) [H, W]                  │
│     - UNet: バイナリスケルトン (thick=3) [H, W]                  │
│                                                                  │
│  f) タプルを返す                                                 │
│     (image, name, points, edges, PAFs, mask, unet, heatmap, id) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. COLLATE_FN - バッチアセンブリ                                 │
├─────────────────────────────────────────────────────────────────┤
│  入力: __getitem__からのタプルのリスト                           │
│                                                                  │
│  処理:                                                           │
│  - Images: リストとして保持 (可変サイズ)                         │
│  - Points, Edges: リストとして保持 (可変グラフサイズ)            │
│  - PAFs: スタック → [B, 2, H, W], [-1, 1]にクランプ             │
│  - Masks: スタック → [B, 1, H, W]                               │
│  - UNet: スタック → [B, 1, H, W], [0, 1]にクランプ              │
│  - Heatmaps: スタック → [B, 1, H, W], [0, 1]にクランプ          │
│                                                                  │
│  出力: [images, points, edges, PAFs, masks, unet,               │
│         heatmaps, ids]                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. モデルのフォワードパス                                         │
├─────────────────────────────────────────────────────────────────┤
│  TreeFormerモデルがバッチを受け取り処理                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 グラフセグメンテーションプロセス

データセットは、PAF/マスク生成のためにNetworkXを使用して木のセグメントを生成します：

```
入力: points, edges
         ↓
    NetworkXグラフを構築
         ↓
    分岐ノード (次数 > 2) を識別
    終端ノード (次数 = 1、ルート以外) を識別
         ↓
    ルートノード (0) からのDFSでセグメントを抽出
    - 分岐ノードで停止
    - 終端ノードで停止
         ↓
    角度でセグメントをソート (木の順序を維持)
         ↓
    セグメントからPAFs、マスク、ヒートマップを生成
```

**例:**
```
木構造:
       0
      / \
     1   2
    / \   \
   3   4   5

エッジ: [[0,1], [0,2], [1,3], [1,4], [2,5]]
分岐ノード: [0, 1]
終端ノード: [3, 4, 5]

セグメント:
1. [0, 1]        # ルートから最初の分岐まで
2. [1, 3]        # 分岐から葉まで
3. [1, 4]        # 分岐から葉まで
4. [0, 2, 5]     # ルートから分岐を経て葉まで
```

---

## 5. カスタム訓練データの作成

### 5.1 前提条件

**必要なライブラリ:**
```python
import torch
import numpy as np
import cv2
from pathlib import Path
```

**必要なデータ:**
1. 入力画像 (PNG形式)
2. 木のスケルトンアノテーション (ノード + エッジ)

### 5.2 データアノテーションの要件

各植物/木の画像について、次のアノテーションが必要です：

1. **ノード位置**: スケルトンに沿ったキーポイント
   - ルートノードはインデックス0である必要がある
   - ピクセル空間の座標

2. **エッジ接続**: 親子関係
   - 木構造を形成 (連結、非巡回)
   - ルートから葉への有向エッジ

**アノテーションガイドライン:**
- 植物の根/基部から開始
- 次の位置にノードを配置:
  - ルートポイント
  - 分岐接合部
  - 分岐の端点/先端
  - 長い分岐に沿った定期的な間隔
- ノードを接続して木構造を形成
- サイクルが存在しないことを確認

### 5.3 ステップバイステップのデータ作成

#### ステップ 1: ディレクトリ構造を準備

```python
import os
from pathlib import Path

def create_dataset_structure(base_path):
    """必要なディレクトリ構造を作成。"""
    splits = ['train', 'val', 'test']
    subdirs = ['data', 'img', 'unet', 'check']

    for split in splits:
        for subdir in subdirs:
            path = Path(base_path) / split / subdir
            path.mkdir(parents=True, exist_ok=True)
            print(f"作成: {path}")

# 使用方法
create_dataset_structure('guyot_data')
```

#### ステップ 2: 画像にアノテーション

**手動アノテーション (推奨ツール):**
- **LabelMe**: ポリゴン/ポイントアノテーション用
- **VGG Image Annotator (VIA)**: ポイントアノテーション用
- **カスタムGUI**: matplotlibまたはOpenCVを使用

**アノテーション形式 (JSON例):**
```json
{
  "image_name": "Set02_IMG_3468.png",
  "image_size": [1920, 1080],
  "nodes": [
    {"id": 0, "x": 960, "y": 1000, "type": "root"},
    {"id": 1, "x": 950, "y": 800, "type": "junction"},
    {"id": 2, "x": 970, "y": 800, "type": "junction"},
    {"id": 3, "x": 940, "y": 600, "type": "tip"},
    {"id": 4, "x": 960, "y": 600, "type": "tip"},
    {"id": 5, "x": 980, "y": 600, "type": "tip"}
  ],
  "edges": [
    [0, 1],
    [0, 2],
    [1, 3],
    [1, 4],
    [2, 5]
  ]
}
```

#### ステップ 3: アノテーションを.pt形式に変換

```python
import torch
import numpy as np
import json
import networkx as nx
from pathlib import Path

class DataPointObject:
    """.ptファイルに保存するオブジェクト。"""
    def __init__(self, points, edges):
        self.list_DETR_points_left_up = points  # torch.Tensor [N, 2]
        self.DETR_node_collections = edges       # torch.Tensor [E, 2]

def validate_tree_structure(edges):
    """エッジが有効な木を形成することを検証。"""
    G = nx.Graph()
    G.add_edges_from(edges.tolist())

    if not nx.is_tree(G):
        raise ValueError("エッジリストが有効な木を形成していません!")

    if not nx.is_connected(G):
        raise ValueError("グラフが連結していません!")

    return True

def create_pt_file(annotation_json_path, image_path, output_data_path, output_img_path):
    """
    アノテーションJSONをTreeFormer .pt形式に変換。

    Args:
        annotation_json_path: JSONアノテーションファイルへのパス
        image_path: ソース画像へのパス
        output_data_path: .ptファイルの出力ディレクトリ
        output_img_path: 画像の出力ディレクトリ
    """
    # アノテーションを読み込む
    with open(annotation_json_path, 'r') as f:
        annotation = json.load(f)

    image_name = annotation['image_name']
    base_name = Path(image_name).stem
    image_size = annotation['image_size']  # [width, height]
    width, height = image_size

    # ノードを抽出し座標を正規化
    nodes = annotation['nodes']
    num_nodes = len(nodes)

    # 一貫した順序を保証するためにIDでノードをソート
    nodes = sorted(nodes, key=lambda x: x['id'])

    # 正規化座標配列を作成
    points_array = np.zeros((num_nodes, 2), dtype=np.float32)
    for node in nodes:
        idx = node['id']
        x = node['x']
        y = node['y']

        # [0, 1]に正規化
        points_array[idx, 0] = x / width
        points_array[idx, 1] = y / height

    # テンソルに変換
    points_tensor = torch.tensor(points_array, dtype=torch.float32)

    # エッジを抽出
    edges = annotation['edges']
    edges_array = np.array(edges, dtype=np.int64)
    edges_tensor = torch.tensor(edges_array, dtype=torch.long)

    # 木構造を検証
    try:
        validate_tree_structure(edges_tensor)
        print(f"✓ {image_name}の有効な木構造")
    except ValueError as e:
        print(f"✗ {image_name}の無効な木構造: {e}")
        return False

    # データオブジェクトを作成
    datapoint = DataPointObject(points_tensor, edges_tensor)

    # .ptファイルを保存
    pt_filename = f"{base_name}.pt"
    pt_path = Path(output_data_path) / pt_filename
    torch.save(datapoint, pt_path)
    print(f"保存: {pt_path}")

    # 画像を出力ディレクトリにコピー
    import shutil
    img_filename = f"{base_name}.png"
    img_output_path = Path(output_img_path) / img_filename
    shutil.copy(image_path, img_output_path)
    print(f"コピー: {img_output_path}")

    return True

# 使用例
annotation_file = "annotations/Set02_IMG_3468.json"
image_file = "images/Set02_IMG_3468.png"
output_data_dir = "guyot_data/train/data"
output_img_dir = "guyot_data/train/img"

create_pt_file(annotation_file, image_file, output_data_dir, output_img_dir)
```

#### ステップ 4: バッチ変換スクリプト

```python
import json
from pathlib import Path
from tqdm import tqdm

def batch_convert_annotations(
    annotation_dir,
    image_dir,
    output_base_dir,
    split='train'
):
    """
    すべてのアノテーションを.pt形式にバッチ変換。

    Args:
        annotation_dir: JSONアノテーションを含むディレクトリ
        image_dir: ソース画像を含むディレクトリ
        output_base_dir: 出力のベースディレクトリ (例: 'guyot_data')
        split: 'train', 'val', または 'test'
    """
    annotation_dir = Path(annotation_dir)
    image_dir = Path(image_dir)
    output_data_dir = Path(output_base_dir) / split / 'data'
    output_img_dir = Path(output_base_dir) / split / 'img'

    # 出力ディレクトリを作成
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=True, exist_ok=True)

    # すべてのアノテーションファイルを取得
    annotation_files = list(annotation_dir.glob('*.json'))

    print(f"{len(annotation_files)}個のアノテーションファイルを検出")

    success_count = 0
    fail_count = 0

    for ann_file in tqdm(annotation_files, desc=f"{split}を変換中"):
        with open(ann_file, 'r') as f:
            annotation = json.load(f)

        image_name = annotation['image_name']
        image_path = image_dir / image_name

        if not image_path.exists():
            print(f"警告: 画像が見つかりません: {image_path}")
            fail_count += 1
            continue

        try:
            result = create_pt_file(
                ann_file,
                image_path,
                output_data_dir,
                output_img_dir
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"{ann_file}の処理中にエラー: {e}")
            fail_count += 1

    print(f"\n変換完了!")
    print(f"  成功: {success_count}")
    print(f"  失敗: {fail_count}")
    print(f"  合計: {len(annotation_files)}")

# 使用方法
batch_convert_annotations(
    annotation_dir='annotations/train',
    image_dir='images/train',
    output_base_dir='guyot_data',
    split='train'
)

batch_convert_annotations(
    annotation_dir='annotations/val',
    image_dir='images/val',
    output_base_dir='guyot_data',
    split='val'
)

batch_convert_annotations(
    annotation_dir='annotations/test',
    image_dir='images/test',
    output_base_dir='guyot_data',
    split='test'
)
```

### 5.4 アノテーションツール (シンプルなインタラクティブ版)

```python
import cv2
import numpy as np
import json
from pathlib import Path

class SimpleTreeAnnotator:
    """
    木構造をアノテーションするためのシンプルなインタラクティブツール。

    コントロール:
    - 左クリック: ノードを追加
    - 右クリック: 最後のノードを削除
    - 'e': エッジモードに入る (2つのノードをクリックして接続)
    - 'u': 最後のエッジを元に戻す
    - 's': アノテーションを保存
    - 'q': 終了
    """

    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        self.display_image = self.image.copy()

        self.height, self.width = self.image.shape[:2]

        self.nodes = []  # (x, y)タプルのリスト
        self.edges = []  # [node_idx1, node_idx2]のリスト

        self.edge_mode = False
        self.edge_start = None

        self.window_name = 'Tree Annotator'

    def draw_annotations(self):
        """現在のアノテーションで画像を再描画。"""
        self.display_image = self.image.copy()

        # エッジを描画
        for edge in self.edges:
            pt1 = tuple(map(int, self.nodes[edge[0]]))
            pt2 = tuple(map(int, self.nodes[edge[1]]))
            cv2.line(self.display_image, pt1, pt2, (0, 255, 0), 2)

        # ノードを描画
        for idx, (x, y) in enumerate(self.nodes):
            color = (0, 0, 255) if idx == 0 else (255, 0, 0)
            cv2.circle(self.display_image, (int(x), int(y)), 5, color, -1)
            cv2.putText(self.display_image, str(idx),
                       (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # モードを表示
        mode_text = "エッジモード" if self.edge_mode else "ノードモード"
        cv2.putText(self.display_image, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(self.window_name, self.display_image)

    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベントを処理。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.edge_mode:
                # エッジのノードを選択
                for idx, (nx, ny) in enumerate(self.nodes):
                    dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                    if dist < 10:
                        if self.edge_start is None:
                            self.edge_start = idx
                            print(f"エッジ開始: ノード {idx}")
                        else:
                            if self.edge_start != idx:
                                self.edges.append([self.edge_start, idx])
                                print(f"エッジを追加: {self.edge_start} -> {idx}")
                            self.edge_start = None
                            self.edge_mode = False
                        break
            else:
                # ノードを追加
                self.nodes.append((x, y))
                print(f"ノード {len(self.nodes)-1} を ({x}, {y}) に追加")

            self.draw_annotations()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 最後のノードを削除
            if self.nodes:
                removed = self.nodes.pop()
                print(f"{removed}のノードを削除")
                # このノードに接続されたエッジを削除
                self.edges = [e for e in self.edges
                             if len(self.nodes)-1 not in e]
                self.draw_annotations()

    def run(self):
        """アノテーションツールを実行。"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("木アノテーションツール")
        print("====================")
        print("コントロール:")
        print("  左クリック: ノードを追加")
        print("  右クリック: 最後のノードを削除")
        print("  'e': エッジモードに入る (2つのノードをクリック)")
        print("  'u': 最後のエッジを元に戻す")
        print("  's': アノテーションを保存")
        print("  'q': 終了")
        print("\n注: 最初のノード (赤) がルートです!")

        self.draw_annotations()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('e'):
                self.edge_mode = True
                self.edge_start = None
                print("エッジモードに入る - 接続する2つのノードをクリック")
                self.draw_annotations()
            elif key == ord('u'):
                if self.edges:
                    removed = self.edges.pop()
                    print(f"エッジを削除: {removed}")
                    self.draw_annotations()
            elif key == ord('s'):
                self.save_annotation()

        cv2.destroyAllWindows()

    def save_annotation(self):
        """アノテーションをJSONファイルに保存。"""
        if len(self.nodes) < 2:
            print("エラー: 少なくとも2つのノードが必要です!")
            return

        if len(self.edges) < 1:
            print("エラー: 少なくとも1つのエッジが必要です!")
            return

        annotation = {
            'image_name': self.image_path.name,
            'image_size': [self.width, self.height],
            'nodes': [
                {
                    'id': idx,
                    'x': int(x),
                    'y': int(y),
                    'type': 'root' if idx == 0 else 'node'
                }
                for idx, (x, y) in enumerate(self.nodes)
            ],
            'edges': self.edges
        }

        output_path = self.image_path.stem + '_annotation.json'
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)

        print(f"アノテーションを{output_path}に保存")
        print(f"  ノード: {len(self.nodes)}")
        print(f"  エッジ: {len(self.edges)}")

# 使用方法
if __name__ == '__main__':
    annotator = SimpleTreeAnnotator('path/to/image.png')
    annotator.run()
```

### 5.5 検証スクリプト

```python
import torch
import networkx as nx
from pathlib import Path

def validate_dataset(data_dir):
    """
    データセットディレクトリ内のすべての.ptファイルを検証。

    Args:
        data_dir: .ptファイルを含む'data'ディレクトリへのパス
    """
    data_dir = Path(data_dir)
    pt_files = list(data_dir.glob('*.pt'))

    print(f"{data_dir}内の{len(pt_files)}個のファイルを検証中")
    print("=" * 60)

    valid_count = 0
    invalid_count = 0

    for pt_file in pt_files:
        try:
            # データを読み込む
            datapoint = torch.load(pt_file)

            # 属性をチェック
            if not hasattr(datapoint, 'list_DETR_points_left_up'):
                print(f"✗ {pt_file.name}: list_DETR_points_left_upが欠落")
                invalid_count += 1
                continue

            if not hasattr(datapoint, 'DETR_node_collections'):
                print(f"✗ {pt_file.name}: DETR_node_collectionsが欠落")
                invalid_count += 1
                continue

            points = datapoint.list_DETR_points_left_up
            edges = datapoint.DETR_node_collections

            # 形状をチェック
            if points.dim() != 2 or points.shape[1] != 2:
                print(f"✗ {pt_file.name}: 無効なポイント形状 {points.shape}")
                invalid_count += 1
                continue

            if edges.dim() != 2 or edges.shape[1] != 2:
                print(f"✗ {pt_file.name}: 無効なエッジ形状 {edges.shape}")
                invalid_count += 1
                continue

            # 座標範囲をチェック
            if (points < 0).any() or (points > 1).any():
                print(f"✗ {pt_file.name}: ポイントが[0,1]範囲外")
                invalid_count += 1
                continue

            # 木構造をチェック
            G = nx.Graph()
            G.add_edges_from(edges.tolist())

            if not nx.is_tree(G):
                print(f"✗ {pt_file.name}: 有効な木構造ではありません")
                invalid_count += 1
                continue

            # ルートノード (0) の存在をチェック
            if 0 not in G.nodes():
                print(f"✗ {pt_file.name}: ルートノード (0) が見つかりません")
                invalid_count += 1
                continue

            # すべてのチェックを通過
            print(f"✓ {pt_file.name}: 有効 "
                  f"(nodes={points.shape[0]}, edges={edges.shape[0]})")
            valid_count += 1

        except Exception as e:
            print(f"✗ {pt_file.name}: エラー - {e}")
            invalid_count += 1

    print("=" * 60)
    print(f"検証完了:")
    print(f"  有効: {valid_count}")
    print(f"  無効: {invalid_count}")
    print(f"  合計: {len(pt_files)}")

    return valid_count, invalid_count

# 使用方法
validate_dataset('guyot_data/train/data')
validate_dataset('guyot_data/val/data')
validate_dataset('guyot_data/test/data')
```

---

## 6. コード例

### 6.1 .ptファイルの読み込みと検査

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# データを読み込む
datapoint = torch.load('guyot_data/train/data/Set02_IMG_3468.pt')

# 属性にアクセス
points = datapoint.list_DETR_points_left_up  # [N, 2]
edges = datapoint.DETR_node_collections       # [E, 2]

print(f"ノード数: {points.shape[0]}")
print(f"エッジ数: {edges.shape[0]}")
print(f"\n最初の5ノード (正規化):")
print(points[:5])
print(f"\n最初の5エッジ:")
print(edges[:5])

# 対応する画像を読み込む
image = plt.imread('guyot_data/train/img/Set02_IMG_3468.png')
height, width = image.shape[:2]

# 正規化座標をピクセルに変換
pixel_coords = points * torch.tensor([width, height])

print(f"\n最初の5ノード (ピクセル座標):")
print(pixel_coords[:5])
```

### 6.2 アノテーションの可視化

```python
import matplotlib.pyplot as plt
import networkx as nx
import torch

def visualize_annotation(image_path, pt_path):
    """木のスケルトンオーバーレイで画像を可視化。"""
    # 画像を読み込む
    image = plt.imread(image_path)
    height, width = image.shape[:2]

    # アノテーションを読み込む
    datapoint = torch.load(pt_path)
    points = datapoint.list_DETR_points_left_up
    edges = datapoint.DETR_node_collections

    # ピクセル座標に変換
    pixel_coords = points.numpy() * np.array([width, height])

    # 図を作成
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # エッジを描画
    for edge in edges:
        start_idx, end_idx = edge
        start = pixel_coords[start_idx]
        end = pixel_coords[end_idx]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                'g-', linewidth=2, alpha=0.7)

    # ノードを描画
    ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
              c='red', s=50, zorder=10)

    # ルートをハイライト
    ax.scatter([pixel_coords[0, 0]], [pixel_coords[0, 1]],
              c='blue', s=100, zorder=11, marker='*')

    # ノードラベルを追加
    for idx, coord in enumerate(pixel_coords):
        ax.text(coord[0] + 10, coord[1] - 10, str(idx),
               color='white', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.set_title(f'木のスケルトンアノテーション\n'
                f'ノード: {len(points)}, エッジ: {len(edges)}')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

# 使用方法
visualize_annotation(
    'guyot_data/train/img/Set02_IMG_3468.png',
    'guyot_data/train/data/Set02_IMG_3468.pt'
)
```

### 6.3 ゼロからシンプルなデータセットを作成

```python
import torch
import numpy as np
from pathlib import Path

class SimpleTreeDataCreator:
    """シンプルな合成木データを作成するためのヘルパークラス。"""

    @staticmethod
    def create_linear_tree(num_nodes=5):
        """シンプルな線形木を作成 (分岐なし)。"""
        # 垂直線に沿ったノード
        points = torch.zeros(num_nodes, 2)
        points[:, 0] = 0.5  # x = 0.5 (中央)
        points[:, 1] = torch.linspace(0.1, 0.9, num_nodes)  # yが変化

        # エッジ: 0->1, 1->2, 2->3, など
        edges = torch.zeros(num_nodes - 1, 2, dtype=torch.long)
        for i in range(num_nodes - 1):
            edges[i] = torch.tensor([i, i + 1])

        return points, edges

    @staticmethod
    def create_branching_tree():
        """1つの分岐点を持つ木を作成。"""
        #      0
        #      |
        #      1
        #     / \
        #    2   3
        #    |   |
        #    4   5

        points = torch.tensor([
            [0.50, 0.90],  # 0: ルート
            [0.50, 0.70],  # 1: 分岐点
            [0.35, 0.50],  # 2: 左分岐
            [0.65, 0.50],  # 3: 右分岐
            [0.35, 0.30],  # 4: 左先端
            [0.65, 0.30],  # 5: 右先端
        ], dtype=torch.float32)

        edges = torch.tensor([
            [0, 1],  # ルートから分岐へ
            [1, 2],  # 分岐から左へ
            [1, 3],  # 分岐から右へ
            [2, 4],  # 左から先端へ
            [3, 5],  # 右から先端へ
        ], dtype=torch.long)

        return points, edges

    @staticmethod
    def save_tree(points, edges, output_dir, name):
        """木データを.ptファイルに保存。"""
        # データオブジェクトを作成
        class DataPoint:
            def __init__(self, pts, edg):
                self.list_DETR_points_left_up = pts
                self.DETR_node_collections = edg

        datapoint = DataPoint(points, edges)

        # 保存
        output_path = Path(output_dir) / f"{name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(datapoint, output_path)

        print(f"保存: {output_path}")
        return output_path

# 使用方法: サンプルデータを作成
creator = SimpleTreeDataCreator()

# 線形木
points1, edges1 = creator.create_linear_tree(num_nodes=10)
creator.save_tree(points1, edges1, 'guyot_data/train/data', 'example_linear')

# 分岐木
points2, edges2 = creator.create_branching_tree()
creator.save_tree(points2, edges2, 'guyot_data/train/data', 'example_branch')
```

### 6.4 データローダーのテスト

```python
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/user/TreeFormer')
from train_mst import LoadCNNDataset, custom_collate_fn

# データセットを作成
dataset = LoadCNNDataset(
    parent_path='guyot_data/train',
    max_size=512,
    is_train=False,
    is_rotate=False
)

print(f"データセットサイズ: {len(dataset)}")

# データローダーを作成
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=0  # デバッグ用
)

# 1バッチをテスト
batch = next(iter(dataloader))
images, points, edges, PAFs, masks, unet, heatmaps, ids = batch

print(f"\nバッチの内容:")
print(f"  画像: {len(images)}枚の画像")
print(f"    - 画像0の形状: {images[0].shape}")
print(f"  ポイント: {len(points)}個のグラフ")
print(f"    - グラフ0のノード: {points[0].shape}")
print(f"  エッジ: {len(edges)}個のグラフ")
print(f"    - グラフ0のエッジ: {edges[0].shape}")
print(f"  PAFsの形状: {PAFs.shape}")
print(f"  マスクの形状: {masks.shape}")
print(f"  UNetの形状: {unet.shape}")
print(f"  ヒートマップの形状: {heatmaps.shape}")
print(f"  ID: {ids}")
```

---

## 7. 重要な実装の詳細

### 7.1 重要な制約

1. **木構造の要件:**
   - 連結である必要がある (単一コンポーネント)
   - 非巡回である必要がある (ループなし)
   - ノード0はルートである必要がある
   - すべてのノードはルートから到達可能である必要がある

2. **座標系:**
   - ポイントは**正規化**座標 [0, 1] で保存される
   - 画像の寸法で正規化: `(x/width, y/height)`
   - 可視化/PAF生成のためにピクセルに変換する必要がある

3. **ファイル命名:**
   - .ptと.pngファイルは一致するベース名が必要
   - 例: `Set02_IMG_3468.pt` ↔ `Set02_IMG_3468.png`

4. **データ型:**
   - ポイント: `torch.float32`
   - エッジ: `torch.long`

### 7.2 よくある落とし穴

1. **間違ったオブジェクト形式:**
   ```python
   # 間違い: 辞書として保存
   data = {'points': points, 'edges': edges}
   torch.save(data, 'file.pt')

   # 正しい: 属性を持つオブジェクトとして保存
   class DataPoint:
       def __init__(self):
           self.list_DETR_points_left_up = points
           self.DETR_node_collections = edges
   torch.save(DataPoint(), 'file.pt')
   ```

2. **座標の正規化を忘れる:**
   ```python
   # 間違い: ピクセル座標
   points = torch.tensor([[960, 540], [970, 530]])

   # 正しい: 正規化座標
   points = torch.tensor([[960/1920, 540/1080],
                          [970/1920, 530/1080]])
   ```

3. **無効な木構造:**
   ```python
   # 間違い: サイクルを作成
   edges = torch.tensor([[0,1], [1,2], [2,0]])  # 三角形

   # 正しい: 木構造
   edges = torch.tensor([[0,1], [1,2]])  # 線形
   ```

4. **ルートノードがゼロでない:**
   ```python
   # 間違い: ルートがノード1
   edges = torch.tensor([[1,0], [1,2]])

   # 正しい: ルートがノード0
   edges = torch.tensor([[0,1], [0,2]])
   ```

### 7.3 パフォーマンスに関する考慮事項

1. **初期化時にすべてのグラフデータを読み込み:**
   - 利点: 高速な訓練反復
   - 欠点: 大規模データセットでの高いメモリ使用量
   - 巨大なデータセットの場合、`load_detr_dataset`をオンデマンド読み込みに変更

2. **画像の読み込み:**
   - 画像は`__getitem__`でディスクから読み込まれる
   - I/Oがボトルネックの場合は前処理とキャッシュを検討

3. **補助ターゲットの生成:**
   - PAFs、マスク、ヒートマップはオンザフライで生成される
   - 高速読み込みのために事前計算してキャッシュ可能

### 7.4 拡張ポイント

**新しい拡張を追加するには:**
```python
def _augment_one_sample(self, img, nodes_list):
    # ここにカスタム拡張を追加
    if random.random() < 0.3:
        img, nodes_list = self._my_custom_augmentation(img, nodes_list)

    # ... 既存の拡張 ...
    return [1, img, nodes, 0]
```

**新しい補助ターゲットを追加するには:**
```python
def __getitem__(self, idx):
    # ... 既存のコード ...

    # カスタムターゲット生成を追加
    my_custom_target = self.generate_my_target(points, edges, image_size)

    return (feature_img, label_img_name0,
            points, edges,
            PAFs, mask, unet, heatmap,
            my_custom_target,  # ここに追加
            file_id)
```

**コレート関数を変更するには:**
```python
def custom_collate_fn(batch):
    # 新しいターゲットを含めてアンパック
    (imgs, names, points, edges, PAFs, masks,
     unet, heatmaps, custom_targets, ids) = zip(*batch)

    # カスタムターゲットを処理
    custom_batch = [item for item in custom_targets]

    # 拡張されたバッチを返す
    return [images, points, edges,
            PAFs, masks, unet, heatmaps,
            custom_batch,  # ここに追加
            ids]
```

---

## まとめ

### クイックリファレンス

**データセット構造:**
```
parent_path/
├── data/          # .ptファイル (グラフアノテーション)
└── img/           # .pngファイル (画像)
```

**.ptファイル形式:**
```python
datapoint.list_DETR_points_left_up  # [N, 2] 正規化 float32
datapoint.DETR_node_collections      # [E, 2] エッジリスト long
```

**データローダー:**
```python
from train_mst import LoadCNNDataset, custom_collate_fn

dataset = LoadCNNDataset(
    parent_path='guyot_data/train',
    max_size=512,
    is_train=True,
    is_rotate=False
)

loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=custom_collate_fn,
    num_workers=4
)
```

**カスタムデータの作成:**
1. 画像にアノテーション (木を形成するノード + エッジ)
2. 座標を [0, 1] に正規化
3. 属性を持つオブジェクトを作成
4. `torch.save()`で保存
5. 木構造を検証
6. 正しいディレクトリ構造に配置

---

## 参考文献

**TreeFormerの主要ファイル:**
- `train_mst.py`: データセット実装を含むメイン訓練スクリプト
- `valid_smd_guyot_nx.py`: データセットバリアントを含む検証スクリプト
- `README.md`: データセット構造ドキュメント
- `configs/tree_2D_use_mst_only1.yaml`: 設定ファイル

**外部依存関係:**
- PyTorch: テンソル操作とデータ読み込み
- NetworkX: グラフ/木構造の検証
- OpenCV: 画像処理と拡張
- NumPy: 数値演算
- Matplotlib: 可視化

---

**ドキュメントバージョン:** 1.0
**最終更新日:** 2025-11-14
**作成者:** TreeFormer Dataset Investigation
