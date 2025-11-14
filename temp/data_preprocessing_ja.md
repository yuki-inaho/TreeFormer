# TreeFormer データ前処理と拡張パイプライン

## 目次
1. [概要](#概要)
2. [データセット構造](#データセット構造)
3. [完全な前処理パイプライン](#完全な前処理パイプライン)
4. [データ拡張技術](#データ拡張技術)
5. [アノテーション変換](#アノテーション変換)
6. [補助表現](#補助表現)
7. [バッチ処理のためのCollate関数](#バッチ処理のためのcollate関数)
8. [カスタムデータへの適用](#カスタムデータへの適用)

---

## 概要

TreeFormerは、画像から木構造グラフを抽出するために設計された高度なデータ前処理と拡張パイプラインを使用します。このパイプラインは以下を処理します：
- キーポイントアノテーション付きの樹木/根系画像
- 変換時のグラフ構造保持
- マルチモーダル表現（PAFs、ヒートマップ、マスク）
- 可変サイズのバッチ処理

**主要ファイル:**
- `train_mst.py` / `train_unmst.py`: `LoadCNNDataset` クラスと前処理ロジックを含む
- `epoch.py`: 訓練/検証ループを含む
- `utils.py`: collate関数を含む

---

## データセット構造

### 期待されるディレクトリレイアウト
```
parent_path/
├── data/           # アノテーションファイル (.pt形式)
│   ├── data_1.pt
│   ├── data_2.pt
│   └── ...
└── img/            # 画像ファイル (.png形式)
    ├── data_1.png
    ├── data_2.png
    └── ...
```

### アノテーション形式（.ptファイル）
各 `.pt` ファイルは以下を含みます：
```python
datapoint.list_DETR_points_left_up    # 正規化されたキーポイントのテンソル [N, 2]
datapoint.DETR_node_collections        # エッジ接続のリスト
```

例：
```python
# キーポイント（[0, 1]に正規化）
list_DETR_points_left_up = torch.tensor([
    [0.5, 0.3],    # ノード 0
    [0.4, 0.5],    # ノード 1
    [0.6, 0.7],    # ノード 2
    # ... その他のノード
])

# エッジ接続（ノードコレクション）
DETR_node_collections = [
    [0, 1, 2],     # ノード 0 -> 1 -> 2 のパス
    [2, 3],        # ノード 2 -> 3 の分岐
    # ... その他のパス
]
```

---

## 完全な前処理パイプライン

### ステップバイステップのプロセス

```python
class LoadCNNDataset(Dataset):
    def __init__(self, parent_path, max_size=1000,
                 max_change_light_rate=0.3, is_train=True, is_rotate=False):
        """
        Args:
            parent_path: 'data/' と 'img/' を含む親ディレクトリのパス
            max_size: リサイズ後の画像の最大次元（デフォルト: 1000）
            max_change_light_rate: 明るさ調整の範囲（デフォルト: 0.3）
            is_train: 拡張を適用するかどうか（デフォルト: True）
            is_rotate: 回転拡張を適用するかどうか（デフォルト: False）
        """
```

### パイプラインフロー図

```
画像とアノテーションの読み込み
         ↓
拡張の適用（is_train=Trueの場合）
  ├─→ 明るさ調整（20%の確率）
  ├─→ ガウシアンノイズ（10%の確率）
  └─→ 反転 + 明るさ/ノイズ（70%の確率）
         ↓
テンソルへの変換と正規化
  ├─→ ToTensor()
  └─→ Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
         ↓
リサイズ（半分のサイズ）
         ↓
最大サイズへの制約
         ↓
回転の適用（is_rotate=Trueの場合）
  ├─→ ランダム角度 [-15, 15]度
  ├─→ キーポイントの変換
  ├─→ 境界外のノードの削除
  ├─→ エッジ分岐の境界への拡張
  └─→ 木構造の検証
         ↓
補助表現の生成
  ├─→ PAFs（Part Affinity Fields）
  ├─→ ヒートマップ（ガウシアンブロブ）
  └─→ マスク（ポリライン）
         ↓
バッチアイテムの返却
```

### 詳細な実装

```python
def __getitem__(self, idx):
    # 1. 画像の読み込み
    label_img_name = self.file_list[idx].split(".pt")[0] + ".png"
    plt_img = plt.imread(os.path.join(self.img_path, label_img_name)).astype(np.float32)

    # RGBA画像の処理（RGBに変換）
    if len(plt_img.shape) == 3 and plt_img.shape[2] == 4:
        plt_img = plt_img[:, :, :3]

    height, width, channels = plt_img.shape

    # 2. キーポイントの読み込みと非正規化
    list_DETR_points_left_up_idx = self.list_DETR_points_left_up[idx]
    nodes_list = list_DETR_points_left_up_idx * torch.tensor([width, height])
    nodes_list = nodes_list.numpy()

    # 3. 拡張の適用
    if self.is_train:
        result_list = self._augment_one_sample(plt_img, nodes_list)
        feature_img, nodes = result_list[1], result_list[2]
    else:
        feature_img = plt_img
        nodes = list_DETR_points_left_up_idx

    list_DETR_points_left_up = torch.tensor(nodes, dtype=torch.float)

    # 4. 画像の正規化
    if len(feature_img.shape) == 3 and feature_img.shape[2] == 3:
        transform_feature = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform_feature = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    feature_img = transform_feature(feature_img)

    # 5. リサイズ（半分のサイズ）
    C, height, width = feature_img.shape
    cut_height = height // 2
    cut_width = width // 2
    feature_img = TF.resize(feature_img, size=[cut_height, cut_width])

    # 6. max_sizeへの制約
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

    # 7. 回転の適用（有効な場合）
    if self.is_rotate:
        feature_img, list_DETR_points_left_up, list_DETR_node_collections_idx, M = \
            self._rotate(feature_img, list_DETR_points_left_up, list_DETR_node_collections_idx)

        # 構造がまだ木であることを検証
        G_tree = nx.Graph()
        G_tree.add_edges_from(list_DETR_node_collections_idx.tolist())
        if not nx.is_tree(G_tree):
            # 回転前のバージョンに戻す
            feature_img = old_save_img
            list_DETR_points_left_up = old_save_list_DETR_points_left_up
            list_DETR_node_collections_idx = old_save_list_DETR_node_collections_idx

    # 8. 補助表現の生成
    feature_size = (feature_img.shape[1], feature_img.shape[2])
    PAFs_idx, mask_idx, unet_idx, heatmap_idx = self.generate_PAFs_by_idx(
        list_DETR_node_collections_idx=list_DETR_node_collections_idx,
        list_DETR_points_left_up_idx=list_DETR_points_left_up,
        feature_size=feature_size,
        sigma=3,
        unet_thickness=3,
        mask_thickness=6
    )

    return (feature_img.contiguous(), label_img_name0,
            list_DETR_points_left_up, list_DETR_node_collections_idx,
            PAFs_idx, mask_idx, unet_idx, heatmap_idx,
            self.ids1[idx])
```

---

## データ拡張技術

### 1. ガウシアンノイズ追加

センサーノイズをシミュレートするために、ガウス分布からサンプリングされたランダムノイズを追加します。

```python
def _gasuss_noise(self, image, mu=0.0, sigma=0.1):
    """
    画像にガウシアンノイズを追加する。

    Args:
        image: 入力画像（[0, 1]に正規化）
        mu: ガウス分布の平均（デフォルト: 0.0）
        sigma: 標準偏差（デフォルト: 0.1）

    Returns:
        ガウシアンノイズを追加した画像、[0, 1]にクリップ
    """
    gasuss_img = copy.deepcopy(image)
    gasuss_img = gasuss_img.astype(np.float32)

    # ノイズの生成
    noise = np.random.normal(mu, sigma, gasuss_img.shape)

    # ノイズを追加してクリップ
    gauss_noise = gasuss_img + noise
    gauss_noise = np.clip(gauss_noise, 0.0, 1.0)

    return gauss_noise

def _addNoise(self, img):
    """ノイズ追加のラッパー。"""
    return self._gasuss_noise(img)
```

**使用例:**
```python
# 画像に適用
noisy_image = dataset._addNoise(your_image)
```

---

### 2. 明るさ/ガンマ調整

ガンマ補正を使用して画像の明るさを調整します。

```python
def _changeLight(self, img):
    """
    ガンマ補正を使用して画像の明るさを調整する。

    Args:
        img: 入力画像

    Returns:
        明るさを調整した画像

    ガンマ値は次の範囲からランダムにサンプリング：
        [1 - max_change_light_rate, 1 + max_change_light_rate]

    gamma > 1: 画像を暗くする
    gamma < 1: 画像を明るくする
    """
    from skimage import exposure

    # max_change_light_rate=0.3の場合、[0.7, 1.3]のランダムガンマ
    flag = random.uniform(
        1 - self.max_change_light_rate,
        1 + self.max_change_light_rate
    )

    light_img = copy.deepcopy(img)
    return exposure.adjust_gamma(light_img, flag)
```

**使用例:**
```python
# 異なる明るさ範囲でデータセットを作成
dataset = LoadCNNDataset(
    parent_path='./data',
    max_change_light_rate=0.3  # ±30%の明るさ調整
)

# 画像に適用
adjusted_image = dataset._changeLight(your_image)
```

---

### 3. 水平反転

画像とキーポイントを水平方向に反転します。

```python
def _flip2(self, img, nodes_list):
    """
    画像とキーポイントを水平方向に反転する。

    Args:
        img: 入力画像（H, W, C）
        nodes_list: キーポイントのリスト [(x, y), ...]

    Returns:
        反転された画像と変換されたキーポイント
    """
    flip_nodes_list = copy.deepcopy(nodes_list)
    flip_img = copy.deepcopy(img)
    w = flip_img.shape[1]

    # 画像を反転（1 = 水平、0 = 垂直）
    img2 = cv2.flip(flip_img, 1)

    # キーポイントの変換
    flip_new_nodes_list = list()
    for x, y in flip_nodes_list:
        flip_new_nodes_list.append([w - x, y])

    return img2, flip_new_nodes_list
```

**変換:**
```
元のポイント: (x, y)
反転後のポイント: (width - x, y)
```

**使用例:**
```python
# 水平反転を適用
flipped_img, flipped_keypoints = dataset._flip2(img, keypoints)
```

---

### 4. グラフ構造保持付き回転

最も洗練された拡張 - 木構造を保持しながら画像を回転します。

```python
def _rotate(self, img, nodes_tensor, connect_tensor):
    """
    画像を回転し、グラフ構造を変換する。

    Args:
        img: 入力画像テンソル（C, H, W）
        nodes_tensor: 正規化されたキーポイント [N, 2]
        connect_tensor: エッジ接続 [E, 2]

    Returns:
        回転された画像、変換されたキーポイント、更新されたエッジ、変換行列

    プロセス:
        1. [-15, 15]度のランダムな回転角度を生成
        2. 画像にアフィン変換を適用
        3. アフィン行列を使用してすべてのキーポイントを変換
        4. 画像境界外のノードを削除
        5. エッジ分岐を画像境界に拡張
        6. 木構造を検証
    """
    rotate_nodes_tensor = copy.deepcopy(nodes_tensor)
    rotate_img = copy.deepcopy(img).cpu().numpy()
    C, height, width = rotate_img.shape

    # (C, H, W) から (H, W, C) に変換
    rotate_img = np.transpose(rotate_img, (1, 2, 0))

    # ランダム回転角度
    angle = random.randint(-15, 15)

    # 回転行列の取得
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # 画像に回転を適用
    img2 = cv2.warpAffine(
        rotate_img, M, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # キーポイントの変換
    rotate_new_nodes_list = list()
    for x, y in (rotate_nodes_tensor * torch.tensor([width, height])).cpu().numpy():
        x1 = M[0][0] * x + M[0][1] * y + M[0][2]
        y1 = M[1][0] * x + M[1][1] * y + M[1][2]
        rotate_new_nodes_list.append([int(x1), int(y1)])

    # グラフの構築
    G = nx.Graph()
    for i, point in enumerate(rotate_new_nodes_list):
        G.add_node(i, point=point)

    for connection in connect_tensor.cpu().numpy().tolist():
        start, end = connection
        if start in G.nodes and end in G.nodes:
            G.add_edge(start, end)

    # 境界外のノードを削除
    for node in rotate_new_nodes_list:
        x, y = node
        if not (0 <= x < width and 0 <= y < height):
            index_id = rotate_new_nodes_list.index(node)
            if index_id in G.nodes:
                G.remove_node(index_id)

    # 残りのノードを抽出
    nodes_data = [G.nodes[node]['point'] for node in G.nodes]

    # 正規化してテンソルに変換
    final_nodes_tensor = torch.tensor(nodes_data, dtype=torch.float32) / \
                        torch.tensor([width, height], dtype=torch.float32)

    # エッジインデックスの更新
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(G.nodes)}
    updated_edges = [(index_mapping[start], index_mapping[end]) for start, end in G.edges]
    final_connect_tensor = torch.tensor(updated_edges, dtype=torch.long)

    # 画像を (C, H, W) に戻す
    img2 = np.transpose(img2, (2, 0, 1))
    img2_tensor = torch.tensor(img2, dtype=torch.float32)

    # エッジ分岐を境界に拡張
    final_nodes_tensor, final_connect_tensor = self._add_edge_branch(
        img, nodes_tensor, connect_tensor,
        final_nodes_tensor, final_connect_tensor, M
    )

    return img2_tensor, final_nodes_tensor, final_connect_tensor, M
```

**回転行列:**
```
M = [cos(θ)  -sin(θ)  tx]
    [sin(θ)   cos(θ)  ty]

変換されたポイント:
x' = M[0,0] * x + M[0,1] * y + M[0,2]
y' = M[1,0] * x + M[1,1] * y + M[1,2]
```

**使用例:**
```python
# 訓練中に回転を有効にする
dataset = LoadCNNDataset(
    parent_path='./data',
    is_train=True,
    is_rotate=True  # 回転を有効化
)

# 回転は__getitem__で自動的に適用される
```

---

### 5. 拡張オーケストレーション

異なる技術を確率的に組み合わせる主要な拡張関数。

```python
def _augment_one_sample(self, check_img, nodes_list):
    """
    確率的選択で拡張を適用する。

    拡張確率:
    - 20%: 明るさ調整のみ
    - 10%: ガウシアンノイズのみ
    - 70%: 反転との組み合わせ
        - 56%: 明るさ + 反転
        - 7%: ノイズ + 反転
        - 7%: 明るさ + ノイズ + 反転

    Args:
        check_img: 入力画像（H, W, C）
        nodes_list: ピクセル座標のキーポイント

    Returns:
        [成功フラグ, 拡張された画像, 正規化されたノード, 0]
    """
    height, width, channels = check_img.shape
    a = random.random()

    if a < 0.2:
        # 20%: 明るさのみ
        crop_img = self._changeLight(check_img)
        nodes_list_check = copy.deepcopy(nodes_list)

    elif 0.2 <= a < 0.3:
        # 10%: ノイズのみ
        crop_img = self._addNoise(check_img)
        nodes_list_check = copy.deepcopy(nodes_list)

    else:
        # 70%: 反転との組み合わせ
        c = random.random()
        if c < 0.8:
            # 56%: 明るさ + 反転
            crop_img = self._changeLight(check_img)
            crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
        elif 0.8 <= c < 0.9:
            # 7%: ノイズ + 反転
            crop_img = self._addNoise(check_img)
            crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)
        else:
            # 7%: 明るさ + ノイズ + 反転
            crop_img = self._changeLight(check_img)
            crop_img = self._addNoise(crop_img)
            crop_img, nodes_list_check = self._flip2(img=crop_img, nodes_list=nodes_list)

    # キーポイントを[0, 1]に正規化
    output_nodes = np.array(nodes_list_check)
    if crop_img.shape[0] == height and crop_img.shape[1] == width:
        output_nodes = output_nodes / np.array([width, height])
        return [1, crop_img, output_nodes, 0]
    else:
        new_height, new_width = crop_img.shape[0], crop_img.shape[1]
        output_nodes = output_nodes / np.array([new_width, new_height])
        new_img = cv2.resize(crop_img, (width, height))
        return [1, new_img, output_nodes, 0]
```

**拡張確率ツリー:**
```
100%
├── 20% → 明るさのみ
├── 10% → ノイズのみ
└── 70% → 反転あり
    ├── 56% → 明るさ + 反転
    ├── 7%  → ノイズ + 反転
    └── 7%  → 明るさ + ノイズ + 反転
```

---

## アノテーション変換

### 座標系

TreeFormerは[0, 1]範囲の正規化座標を使用します：

```python
# ピクセル座標 → 正規化
normalized_x = pixel_x / image_width
normalized_y = pixel_y / image_height

# 正規化 → ピクセル座標
pixel_x = normalized_x * image_width
pixel_y = normalized_y * image_height
```

### 変換の例

#### 1. 反転変換
```python
# 元の座標
point = (0.7, 0.3)  # 正規化座標

# 水平反転後
flipped_point = (1.0 - 0.7, 0.3) = (0.3, 0.3)

# ピクセル空間（width=800）
original_pixel = (560, 240)
flipped_pixel = (800 - 560, 240) = (240, 240)
```

#### 2. 回転変換
```python
# 15度回転の回転行列Mが与えられた場合
M = cv2.getRotationMatrix2D((width/2, height/2), 15, 1)

# ポイント (400, 300) を変換（width=800, height=600）
x_new = M[0,0] * 400 + M[0,1] * 300 + M[0,2]
y_new = M[1,0] * 400 + M[1,1] * 300 + M[1,2]
```

### エッジ分岐の拡張

回転後、画像外に出るエッジ分岐は境界に賢く拡張されます：

```python
def _add_edge_branch(self, img, nodes_tensor, connect_tensor,
                     final_nodes_tensor, final_connect_tensor, M):
    """
    回転後にエッジ分岐を画像境界に拡張する。

    これにより、回転が分岐を画像境界外に押し出した場合の
    情報損失を防ぎます。

    プロセス:
        1. 終端ノード（次数 = 1）を特定
        2. 各終端ノードの子に最も近い画像エッジを見つける
        3. そのエッジとの交点を計算
        4. 有効な場合、交点に新しいノードを追加
        5. グラフ接続を更新

    Args:
        img: 元の画像
        nodes_tensor: 元のキーポイント
        connect_tensor: 元のエッジ
        final_nodes_tensor: 回転されたキーポイント
        final_connect_tensor: 回転されたエッジ
        M: 回転行列

    Returns:
        拡張された分岐を含む更新されたノードとエッジ
    """
    # 画像のエッジを線方程式として定義
    edge_func = {
        'top': [0, 1, 0],           # y = 0
        'bottom': [0, 1, -height],  # y = height
        'left': [1, 0, 0],          # x = 0
        'right': [1, 0, -width]     # x = width
    }

    # ... (実装の詳細はコード内)

    return final_rotate_nodes_tensor, final_rotate_connect_tensor
```

**例:**
```
回転前:                    回転後（拡張あり）:

  ●---●---●                   ●
   \                           \
    ●                           ●---●---●
                                         \
                                          ●---● (境界に拡張)
```

### グラフ構造の検証

```python
# 回転後、木構造を検証
G_tree = nx.Graph()
G_tree.add_edges_from(list_DETR_node_collections_idx.tolist())

if not nx.is_tree(G_tree):
    # 構造が壊れている場合は元に戻す
    feature_img = old_save_img
    list_DETR_points_left_up = old_save_list_DETR_points_left_up
    list_DETR_node_collections_idx = old_save_list_DETR_node_collections_idx
```

**チェックされる木の性質:**
- 連結グラフ
- サイクルなし
- Nノード → N-1エッジ

---

## 補助表現

TreeFormerは訓練を支援するために複数の表現を生成します：

### 1. Part Affinity Fields（PAFs）

エッジの方向と位置をベクトル場としてエンコードします。

```python
def generate_PAFs(height, width, points, paths, line_thickness=2):
    """
    エッジ表現のためのPart Affinity Fieldsを生成する。

    PAFsはグラフ内のエッジの位置と方向の両方をエンコードします。
    エッジに沿った各ピクセルには、エッジに沿って指す単位ベクトルが含まれます。

    Args:
        height, width: 画像の次元
        points: 正規化されたキーポイント
        paths: パスのリスト（ノードインデックスのシーケンス）
        line_thickness: PAF領域の太さ（デフォルト: 2）

    Returns:
        PAFs: 単位ベクトルを含む形状(H, W, 2)の配列

    各エッジ（node_i → node_j）について:
        - 単位方向ベクトルを計算: (ux, uy)
        - ノード間の太い線でこのベクトルを描画
        - PAFs[y, x, :] = (ux, uy) 線内のすべてのピクセルについて
    """
    PAFs = np.zeros((height, width, 2), dtype=np.float32)

    for branch in paths:
        for idx in range(len(branch) - 1):
            start_point = points[branch[idx]]
            end_point = points[branch[idx + 1]]

            # ピクセル座標に変換
            x1, y1 = int(start_point[0] * width), int(start_point[1] * height)
            x2, y2 = int(end_point[0] * width), int(end_point[1] * height)

            # 単位ベクトルを計算
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length == 0:
                continue
            ux = (x2 - x1) / length
            uy = (y2 - y1) / length

            # 線に沿ってPAFを描画
            for t in np.linspace(0, 1, int(length)):
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < width and 0 <= y < height:
                    PAFs[y-line_thickness:y+line_thickness,
                         x-line_thickness:x+line_thickness, 0] = ux
                    PAFs[y-line_thickness:y+line_thickness,
                         x-line_thickness:x+line_thickness, 1] = uy

    return PAFs
```

**視覚的表現:**
```
エッジ付き画像:          PAF可視化:

    ●                        ↗
     \                      ↗
      ●---●              →→→
           \            ↘
            ●          ↘

(矢印はPAFベクトルの方向と大きさを示す)
```

**使用法:**
```python
PAFs = generate_PAFs(
    height=512,
    width=512,
    points=keypoints,  # 正規化 [0, 1]
    paths=sorted_segments,
    line_thickness=2
)

# PAFs.shape = (512, 512, 2)
# PAFs[:, :, 0] = 単位ベクトルのx成分
# PAFs[:, :, 1] = 単位ベクトルのy成分
```

---

### 2. ガウシアンヒートマップ

キーポイントの位置をガウシアンブロブとして表現します。

```python
def generate_heatmap(normalized_kpts, image_size, sigma):
    """
    キーポイント位置特定のためのガウシアンヒートマップを生成する。

    各キーポイントは2Dガウシアンブロブとして表現されます。
    重なり合う複数のガウシアンは最大値プーリングを使用します。

    Args:
        normalized_kpts: [0, 1]範囲のキーポイント
        image_size: (height, width)
        sigma: ガウシアンの標準偏差（ブロブのサイズを制御）

    Returns:
        Heatmap: [0, 1]の値を持つ形状(H, W)の配列

    ガウシアン式:
        G(x, y) = exp(-0.5 * ((x - x_kp)^2 + (y - y_kp)^2) / sigma^2)
    """
    H, W = image_size
    heatmap = np.zeros((H, W))

    for keypoint in normalized_kpts:
        x_normalized, y_normalized = keypoint
        x = x_normalized * W
        y = y_normalized * H

        # メッシュグリッドを作成
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))

        # ガウシアンを計算
        gaussian = np.exp(-0.5 * ((xx - x)**2 + (yy - y)**2) / sigma**2)

        # 小さい値をしきい値処理
        gaussian[gaussian < 0.01] = 0

        # 重なり合うガウシアンに最大値プーリング
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap
```

**視覚的表現:**
```
キーポイント:            ヒートマップ:

   ●                      ▓▓▓
                         ▓███▓
       ●               ▓▓▓███▓▓▓
                              ▓███▓
                              ▓▓▓▓▓

(明るさはガウシアン強度を示す)
```

**Sigmaパラメータの効果:**
```
sigma=1 (狭い):        sigma=3 (中):          sigma=5 (広い):
    ██                      ▓▓▓▓                  ░░▓▓▓▓░░
    ██                    ▓▓████▓▓              ░░▓▓████▓▓░░
                          ▓▓████▓▓            ░░▓▓████████▓▓░░
                            ▓▓▓▓              ░░▓▓████████▓▓░░
                                                ░░▓▓▓▓▓▓░░
```

---

### 3. ポリラインマスク

太い線として木構造を表すバイナリマスク。

```python
def create_mask_with_polylines(image_shape, keypoints, segments, thickness=2):
    """
    木構造のためのポリライン付きバイナリマスクを作成する。

    Args:
        image_shape: (height, width)
        keypoints: 正規化されたキーポイント [N, 2]
        segments: パスのリスト（ノードインデックスのシーケンス）
        thickness: 線の太さ（ピクセル単位）

    Returns:
        木構造に沿って1を持つ形状(H, W)のバイナリマスク

    cv2.polylinesを使用して接続されたセグメントを描画します。
    """
    kpts = copy.deepcopy(keypoints)

    # キーポイントを画像の次元にスケール
    kpts[:, 0] *= image_shape[1]
    kpts[:, 1] *= image_shape[0]

    mask = np.zeros(image_shape, dtype=np.uint8)

    for segment in segments:
        # このセグメントのポイントを抽出
        segment_points = kpts[segment].reshape((-1, 1, 2)).astype(np.int32)

        # ポリラインを描画
        cv2.polylines(
            mask,
            [segment_points],
            isClosed=False,
            color=1,
            thickness=thickness
        )

    return mask
```

**太さの比較:**
```
thickness=2:            thickness=4:            thickness=6:
    ●                       ●                       ●
    ║                      ║║                      ║║║
    ●══●                  ●════●                  ●══════●
        ║                     ║║                     ║║║
        ●                     ●                      ●
```

**複数の表現:**
```python
# generate_PAFs_by_idx()で生成

# 1. 損失計算用のマスク（太い）
PAFs_mask = create_mask_with_polylines(
    orig_size, kpts, segments, thickness=6
)
mask_tensor = torch.tensor(PAFs_mask, dtype=torch.bool)

# 2. UNet補助ターゲット（中）
PAFs_unet = create_mask_with_polylines(
    orig_size, kpts, segments, thickness=2
)
unet_tensor = torch.tensor(PAFs_unet, dtype=torch.float32)
```

---

### 4. グラフセグメンテーション

セグメントはDFS探索を使用して識別されます：

```python
def find_segments_v2(start_node, node_collections, branching_nodes, end_nodes):
    """
    DFSを使用して木内のすべてのパスセグメントを見つける。

    セグメントは以下からのパスです:
        - 開始ノード → 分岐ノード
        - 分岐ノード → 分岐ノード
        - 分岐ノード → 終端ノード
        - 開始ノード → 終端ノード（分岐がない場合）

    Args:
        start_node: 木のルート（通常はノード0）
        node_collections: エッジ接続のリスト
        branching_nodes: 次数 > 2のノード
        end_nodes: 次数 = 1の葉ノード

    Returns:
        セグメントのリスト、各セグメントはノードインデックスのリスト
    """
    segments = []
    visited_nodes = set()

    def dfs(node, path):
        visited_nodes.add(node)
        path.append(node)

        if node in branching_nodes:
            # 分岐点までのパスを保存
            segments.append(path.copy())
            # 分岐ノードから新しいパスを開始
            for collection in node_collections:
                if node in collection:
                    for neighbor in collection:
                        if neighbor not in visited_nodes:
                            dfs(neighbor, [node])
            return

        if node in end_nodes:
            # 葉に到達
            segments.append(path.copy())
            return

        # パスに沿って続ける
        for collection in node_collections:
            if node in collection:
                for neighbor in collection:
                    if neighbor not in visited_nodes:
                        dfs(neighbor, path.copy())

    dfs(start_node, [])
    return segments
```

**セグメンテーションの例:**
```
木構造:
        0 (開始)
        |
        1
       / \
      2   3 (分岐)
     / \   \
    4   5   6

セグメント:
[0, 1, 3]        # 開始 → 分岐
[3, 2]           # 分岐 → 分岐
[2, 4]           # 分岐 → 終端
[2, 5]           # 分岐 → 終端
[3, 6]           # 分岐 → 終端
```

---

## バッチ処理のためのCollate関数

TreeFormerは可変サイズの画像とグラフをカスタムcollate関数を使用して処理します：

```python
def custom_collate_fn(batch):
    """
    可変サイズのサンプルをバッチ処理するためのカスタムcollate関数。

    Args:
        batch: __getitem__からのアイテムのリスト
               各アイテムは (feature_img, label_img_name0,
                           list_DETR_points_left_up, list_DETR_node_collections,
                           PAFs_idx, mask_idx, unet_idx, heatmap_idx, ids1)

    Returns:
        バッチ化されたデータ:
        - images: テンソルのリスト（可変サイズ）
        - points_left_up: キーポイントテンソルのリスト
        - edges: エッジテンソルのリスト
        - PAFs_concatenated: [B, 2, H, W]
        - mask_concatenated: [B, 1, H, W]
        - unet_concatenated: [B, 1, H, W]
        - heatmap_concatenated: [B, 1, H, W]
        - detr_ids: IDのリスト
    """
    (feature_img, label_img_name0, list_DETR_points_left_up, list_DETR_node_collections,
     list_PAFs, list_mask, list_unet, list_heatmap, ids1) = zip(*batch)

    # 数値安定性定数
    ACT_1 = 0.9999999  # クランプの最大値
    ACT_0 = 0.0000001  # クランプの最小値

    # 1. 画像をリストとして保持（可変サイズ）
    images = [item.to(torch.float32) for item in feature_img]

    # 2. キーポイントとエッジをリストとして保持
    points_left_up = [item for item in list_DETR_points_left_up]
    edges = [item for item in list_DETR_node_collections]

    # 3. PAFsを変換して連結
    # PAFs: (H, W, 2) → (1, 2, H, W)
    PAFs_list_transformed = [PAFs.unsqueeze(0).permute(0, 3, 1, 2)
                             for PAFs in list_PAFs]

    # 4. マスクを変換して連結
    # Masks: (H, W) → (1, 1, H, W)
    mask_list_transformed = [mask.unsqueeze(0).unsqueeze(0)
                            for mask in list_mask]
    unet_list_transformed = [unet.unsqueeze(0).unsqueeze(0)
                            for unet in list_unet]
    heatmap_list_transformed = [heatmap.unsqueeze(0).unsqueeze(0)
                               for heatmap in list_heatmap]

    # 5. バッチ次元に沿って連結
    PAFs_concatenated = torch.cat(PAFs_list_transformed, 0)
    mask_concatenated = torch.cat(mask_list_transformed, 0).contiguous()
    unet_concatenated = torch.cat(unet_list_transformed, 0)
    heatmap_concatenated = torch.cat(heatmap_list_transformed, 0)

    # 6. 数値安定性のために値をクランプ
    PAFs_concatenated = torch.clamp(PAFs_concatenated, min=-ACT_1, max=ACT_1)
    unet_concatenated = torch.clamp(unet_concatenated, min=ACT_0, max=ACT_1)
    heatmap_concatenated = torch.clamp(heatmap_concatenated, min=ACT_0, max=ACT_1)

    # 7. IDの準備
    detr_ids = list(ids1)

    return [images, points_left_up, edges,
            PAFs_concatenated, mask_concatenated, unet_concatenated, heatmap_concatenated,
            detr_ids],
```

### バッチ構造

```python
# batch_size=2のバッチの例

batch = {
    'images': [
        torch.Tensor([3, 512, 384]),  # 画像1: 512x384
        torch.Tensor([3, 480, 640]),  # 画像2: 480x640（異なるサイズ！）
    ],

    'points_left_up': [
        torch.Tensor([25, 2]),  # 画像1: 25個のキーポイント
        torch.Tensor([18, 2]),  # 画像2: 18個のキーポイント（異なる数！）
    ],

    'edges': [
        torch.Tensor([24, 2]),  # 画像1: 24個のエッジ
        torch.Tensor([17, 2]),  # 画像2: 17個のエッジ
    ],

    'PAFs_concatenated': torch.Tensor([2, 2, 512, 384]),
    'mask_concatenated': torch.Tensor([2, 1, 512, 384]),
    'unet_concatenated': torch.Tensor([2, 1, 512, 384]),
    'heatmap_concatenated': torch.Tensor([2, 1, 512, 384]),

    'detr_ids': ['data_1', 'data_2']
}
```

### DataLoaderのセットアップ

```python
from torch.utils.data import DataLoader

# 訓練データセット
dataset_train = LoadCNNDataset(
    parent_path='./train',
    max_size=512,
    max_change_light_rate=0.3,
    is_train=True,
    is_rotate=True
)

# 訓練ローダー
train_loader = DataLoader(
    dataset_train,
    batch_size=8,
    shuffle=False,  # DistributedSamplerを使用
    collate_fn=custom_collate_fn,
    drop_last=True,
    pin_memory=True,
    num_workers=4,
    sampler=train_sampler  # DistributedSampler
)

# 検証データセット（拡張なし）
dataset_val = LoadCNNDataset(
    parent_path='./val',
    max_size=512,
    max_change_light_rate=0.3,
    is_train=False,   # 拡張なし
    is_rotate=False   # 回転なし
)

# 検証ローダー
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

## カスタムデータへの適用

### ステップ1: データの準備

#### ディレクトリ構造
```bash
your_dataset/
├── train/
│   ├── data/
│   │   ├── sample_001.pt
│   │   ├── sample_002.pt
│   │   └── ...
│   └── img/
│       ├── sample_001.png
│       ├── sample_002.png
│       └── ...
├── val/
│   ├── data/
│   │   └── ...
│   └── img/
│       └── ...
└── test/
    ├── data/
    │   └── ...
    └── img/
        └── ...
```

#### アノテーションファイルの作成

```python
import torch

class AnnotationData:
    """アノテーションデータのコンテナ。"""
    def __init__(self, keypoints, edges):
        self.list_DETR_points_left_up = keypoints
        self.DETR_node_collections = edges

# 例: 木構造のアノテーションを作成
def create_annotation_example():
    """
    この木のアノテーションを作成:
        0 (ルート)
        |
        1
       / \
      2   3
      |   |
      4   5
    """
    # キーポイント（[0, 1]に正規化）
    keypoints = torch.tensor([
        [0.5, 0.1],   # ノード0: 上部中央のルート
        [0.5, 0.3],   # ノード1: ルートの下
        [0.3, 0.6],   # ノード2: 左の分岐
        [0.7, 0.6],   # ノード3: 右の分岐
        [0.3, 0.9],   # ノード4: 左の葉
        [0.7, 0.9],   # ノード5: 右の葉
    ], dtype=torch.float32)

    # エッジコレクション（木内のパス）
    # 各パスは接続されたノードのシーケンス
    edges = [
        [0, 1, 2, 4],  # パス: ルート → ノード1 → ノード2 → 葉4
        [1, 3, 5],     # パス: ノード1 → ノード3 → 葉5
    ]

    # アノテーションオブジェクトの作成
    annotation = AnnotationData(keypoints, edges)

    # ファイルに保存
    torch.save(annotation, 'sample_001.pt')

    return annotation

# 複数のアノテーションを作成
for i in range(100):
    annotation = create_annotation_for_sample(i)
    torch.save(annotation, f'train/data/sample_{i:03d}.pt')
```

#### アノテーション形式の詳細

```python
# キーポイント形式
keypoints.shape = (N, 2)  # N = ノード数
keypoints.dtype = torch.float32
# [0, 1]範囲の値（正規化）
# keypoints[i] = [x_normalized, y_normalized]

# 例
keypoints = torch.tensor([
    [0.25, 0.30],  # ノード0: 幅の25%、高さの30%
    [0.50, 0.60],  # ノード1: 幅の50%、高さの60%
    [0.75, 0.90],  # ノード2: 幅の75%、高さの90%
])

# エッジ形式（ノードコレクション）
# パスのリスト、各パスはノードインデックスのリスト
edges = [
    [0, 1, 2],     # ノード0→1→2を接続するパス
    [1, 3],        # ノード1→3の分岐
]

# 重要: エッジは有効な木構造を形成する必要があります
# - 連結（すべてのノードがルートから到達可能）
# - 非巡回（ループなし）
# - ルートは通常ノード0
```

---

### ステップ2: データセットの作成

```python
from train_mst import LoadCNNDataset, custom_collate_fn
from torch.utils.data import DataLoader

# 拡張付き訓練データセット
train_dataset = LoadCNNDataset(
    parent_path='./your_dataset/train',
    max_size=512,              # 最大画像次元
    max_change_light_rate=0.3, # ±30%の明るさ
    is_train=True,             # 拡張を有効化
    is_rotate=True             # 回転を有効化
)

# 拡張なし検証データセット
val_dataset = LoadCNNDataset(
    parent_path='./your_dataset/val',
    max_size=512,
    max_change_light_rate=0.3,
    is_train=False,   # 拡張を無効化
    is_rotate=False   # 回転を無効化
)

# テストデータセット
test_dataset = LoadCNNDataset(
    parent_path='./your_dataset/test',
    max_size=512,
    is_train=False,
    is_rotate=False
)

print(f"訓練サンプル数: {len(train_dataset)}")
print(f"検証サンプル数: {len(val_dataset)}")
print(f"テストサンプル数: {len(test_dataset)}")
```

---

### ステップ3: DataLoaderの作成

```python
# 訓練ローダー
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# 検証ローダー
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=4,
    pin_memory=True
)

# テストローダー
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn
)
```

---

### ステップ4: 反復と可視化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_batch_item(batch, idx=0):
    """
    バッチから単一アイテムを可視化する。

    Args:
        batch: DataLoaderからの出力
        idx: 可視化するアイテムのインデックス（デフォルト: 0）
    """
    images, points, edges, PAFs, masks, unets, heatmaps, ids = batch[0]

    # 特定のアイテムを取得
    img = images[idx].cpu().numpy()
    pts = points[idx].cpu().numpy()
    edg = edges[idx].cpu().numpy()
    paf = PAFs[idx].cpu().numpy()
    mask = masks[idx].cpu().numpy()
    heatmap = heatmaps[idx].cpu().numpy()

    # 画像の非正規化
    img = (img * 0.5) + 0.5  # 正規化を逆転
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = np.clip(img, 0, 1)

    # 可視化の作成
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 元の画像
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('元の画像')
    axes[0, 0].axis('off')

    # 2. キーポイント付き画像
    axes[0, 1].imshow(img)
    H, W = img.shape[:2]
    # 正規化座標をピクセル座標に変換
    pixel_pts = pts * np.array([W, H])
    axes[0, 1].scatter(pixel_pts[:, 0], pixel_pts[:, 1],
                       c='red', s=50, marker='o')
    for i, pt in enumerate(pixel_pts):
        axes[0, 1].text(pt[0], pt[1], str(i),
                       color='white', fontsize=8)
    axes[0, 1].set_title('キーポイント')
    axes[0, 1].axis('off')

    # 3. グラフ構造付き画像
    axes[0, 2].imshow(img)
    axes[0, 2].scatter(pixel_pts[:, 0], pixel_pts[:, 1],
                       c='red', s=50, marker='o')
    # エッジを描画
    for edge in edg:
        pt1 = pixel_pts[int(edge[0])]
        pt2 = pixel_pts[int(edge[1])]
        axes[0, 2].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                       'b-', linewidth=2)
    axes[0, 2].set_title('グラフ構造')
    axes[0, 2].axis('off')

    # 4. ヒートマップ
    axes[1, 0].imshow(heatmap.squeeze(), cmap='hot')
    axes[1, 0].set_title('キーポイントヒートマップ')
    axes[1, 0].axis('off')

    # 5. マスク
    axes[1, 1].imshow(mask.squeeze(), cmap='gray')
    axes[1, 1].set_title('木構造マスク')
    axes[1, 1].axis('off')

    # 6. PAF大きさ
    paf_magnitude = np.sqrt(paf[0]**2 + paf[1]**2)
    axes[1, 2].imshow(paf_magnitude, cmap='viridis')
    axes[1, 2].set_title('PAF大きさ')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'visualization_{ids[idx]}.png', dpi=150, bbox_inches='tight')
    plt.show()

# 最初のバッチを可視化
for batch in train_loader:
    visualize_batch_item(batch, idx=0)
    break
```

---

### ステップ5: カスタム拡張設定

```python
class CustomDataset(LoadCNNDataset):
    """修正された拡張付きカスタムデータセット。"""

    def _augment_one_sample(self, check_img, nodes_list):
        """
        拡張関数をオーバーライド。

        カスタム拡張確率:
        - 30%: 明るさのみ
        - 20%: ノイズのみ
        - 50%: 組み合わせ
        """
        height, width, channels = check_img.shape
        a = random.random()

        if a < 0.3:
            # 30%: 明るさ
            crop_img = self._changeLight(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)

        elif 0.3 <= a < 0.5:
            # 20%: ノイズ
            crop_img = self._addNoise(check_img)
            nodes_list_check = copy.deepcopy(nodes_list)

        else:
            # 50%: 組み合わせ
            crop_img = self._changeLight(check_img)
            crop_img = self._addNoise(crop_img)
            crop_img, nodes_list_check = self._flip2(
                img=crop_img,
                nodes_list=nodes_list
            )

        # 正規化
        output_nodes = np.array(nodes_list_check)
        output_nodes = output_nodes / np.array([width, height])

        return [1, crop_img, output_nodes, 0]

# カスタムデータセットを使用
custom_train = CustomDataset(
    parent_path='./your_dataset/train',
    max_size=512,
    is_train=True,
    is_rotate=True
)
```

---

### ステップ6: 高度な使用法 - カスタム拡張の追加

```python
class AdvancedDataset(LoadCNNDataset):
    """追加の拡張技術を持つデータセット。"""

    def _add_blur(self, img):
        """ガウシアンブラーを追加。"""
        import cv2
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def _add_contrast(self, img, alpha=None):
        """コントラストを調整。"""
        if alpha is None:
            alpha = random.uniform(0.7, 1.3)
        return np.clip(img * alpha, 0, 1)

    def _elastic_transform(self, img, nodes_list):
        """弾性変形を適用（高度）。"""
        # 弾性変換の実装
        # キーポイントの慎重な処理が必要
        pass

    def _augment_one_sample(self, check_img, nodes_list):
        """拡張されたパイプライン。"""
        height, width, channels = check_img.shape
        a = random.random()

        # 基本拡張を適用
        if a < 0.2:
            crop_img = self._changeLight(check_img)
            crop_img = self._add_contrast(crop_img)
            nodes_list_check = copy.deepcopy(nodes_list)

        elif 0.2 <= a < 0.4:
            crop_img = self._addNoise(check_img)
            crop_img = self._add_blur(crop_img)
            nodes_list_check = copy.deepcopy(nodes_list)

        else:
            # 複雑な組み合わせ
            crop_img = self._changeLight(check_img)
            crop_img = self._add_contrast(crop_img)

            if random.random() < 0.5:
                crop_img = self._add_blur(crop_img)

            crop_img, nodes_list_check = self._flip2(
                img=crop_img,
                nodes_list=nodes_list
            )

        # 正規化
        output_nodes = np.array(nodes_list_check)
        output_nodes = output_nodes / np.array([width, height])

        return [1, crop_img, output_nodes, 0]

# 高度なデータセットを使用
advanced_train = AdvancedDataset(
    parent_path='./your_dataset/train',
    max_size=512,
    is_train=True,
    is_rotate=True
)
```

---

### ステップ7: 検証とデバッグ

```python
def validate_dataset(dataset, num_samples=10):
    """
    データセットのアノテーションと前処理を検証する。

    チェック項目:
    - 画像が正しく読み込まれる
    - キーポイントが有効な範囲[0, 1]内
    - エッジが有効な木構造を形成
    - NaNまたはInf値がない
    - 拡張がグラフ構造を保持
    """
    import networkx as nx

    issues = []

    for i in range(min(num_samples, len(dataset))):
        try:
            # サンプルを取得
            sample = dataset[i]
            img, name, kpts, edges, paf, mask, unet, heatmap, id_ = sample

            # 画像をチェック
            if torch.isnan(img).any() or torch.isinf(img).any():
                issues.append(f"サンプル {i}: 画像にNaNまたはInfが含まれる")

            # キーポイントの範囲をチェック
            if (kpts < 0).any() or (kpts > 1).any():
                issues.append(f"サンプル {i}: キーポイントが範囲[0, 1]外")

            # 木構造をチェック
            G = nx.Graph()
            G.add_edges_from(edges.numpy().tolist())
            if not nx.is_tree(G):
                issues.append(f"サンプル {i}: エッジが有効な木を形成していない")

            # 切断されたノードをチェック
            if kpts.shape[0] != len(G.nodes):
                issues.append(f"サンプル {i}: キーポイントとグラフノードの不一致")

            # 補助表現をチェック
            if torch.isnan(paf).any():
                issues.append(f"サンプル {i}: PAFにNaNが含まれる")
            if torch.isnan(heatmap).any():
                issues.append(f"サンプル {i}: ヒートマップにNaNが含まれる")

        except Exception as e:
            issues.append(f"サンプル {i}: エラー - {str(e)}")

    # 結果を出力
    if not issues:
        print(f"✓ {num_samples}個のサンプルすべてが正常に検証されました！")
    else:
        print(f"✗ {len(issues)}個の問題が見つかりました:")
        for issue in issues:
            print(f"  - {issue}")

    return issues

# データセットを検証
issues = validate_dataset(train_dataset, num_samples=100)
```

---

## まとめ

### TreeFormer前処理の主要機能

1. **柔軟な画像処理**
   - 可変画像サイズのサポート
   - RGBAからRGBへの自動変換
   - 設定可能な最大サイズ制約

2. **高度な拡張**
   - ガンマ補正による明るさ調整
   - ガウシアンノイズ追加
   - キーポイント変換付き水平反転
   - グラフ構造保持付き回転
   - 技術の確率的組み合わせ

3. **グラフ対応変換**
   - キーポイントとエッジを一緒に変換
   - 境界外のノードを削除
   - エッジ分岐を境界に拡張
   - 回転後の木構造検証

4. **マルチモーダル表現**
   - エッジエンコーディングのためのPart Affinity Fields（PAFs）
   - キーポイント位置特定のためのガウシアンヒートマップ
   - 構造表現のためのバイナリマスク
   - 複数の太さバリアント

5. **効率的なバッチ処理**
   - 可変サイズの画像をリストとして処理
   - サンプルごとに可変数のノード/エッジ
   - 固定サイズの補助表現
   - クランプによる数値安定性

### 設定パラメータ

```python
# 調整する主要パラメータ
LoadCNNDataset(
    parent_path='./data',
    max_size=512,              # 画像サイズ制約（デフォルト: 1000）
    max_change_light_rate=0.3, # 明るさ範囲±30%（デフォルト: 0.3）
    is_train=True,             # 拡張を有効化
    is_rotate=False            # 回転を有効化（デフォルト: False）
)

# 回転パラメータ（_rotate内でハードコード）
angle = random.randint(-15, 15)  # 回転範囲

# 補助表現パラメータ
generate_PAFs_by_idx(
    ...,
    sigma=3,           # ヒートマップガウシアン標準偏差（デフォルト: 3）
    unet_thickness=3,  # UNetマスク太さ（デフォルト: 2）
    mask_thickness=6   # 損失マスク太さ（デフォルト: 6）
)
```

### よくある落とし穴と解決策

| 問題 | 解決策 |
|-------|----------|
| 反転後にキーポイントが境界外 | 反転前にキーポイントを正規化 |
| 回転後に木構造が壊れる | `nx.is_tree()`検証が有効であることを確認 |
| PAFsにNaN | エッジが長さゼロでないことを確認 |
| 大きなバッチでメモリ問題 | batch_sizeまたはmax_sizeパラメータを削減 |
| データ読み込みが遅い | DataLoaderのnum_workersを増やす |
| 拡張が強すぎる | max_change_light_rateを削減、回転を無効化 |

---

## 参考文献

- **主要訓練スクリプト**: `/home/user/TreeFormer/train_mst.py`
- **エポック関数**: `/home/user/TreeFormer/epoch.py`
- **ユーティリティ**: `/home/user/TreeFormer/utils.py`
- **設定例**: `/home/user/TreeFormer/configs/tree_2D_use_mst_only1.yaml`

---

**ドキュメント作成日:** 2025-11-14
**TreeFormerバージョン:** リポジトリコード解析に基づく
**著者:** 自動ドキュメントシステム
