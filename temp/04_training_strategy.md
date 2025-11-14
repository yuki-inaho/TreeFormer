# 訓練戦略

## 概要

TreeFormerの訓練は、標準的なDeep Learning訓練パイプラインに加えて、MST制約という独自の要素を含みます。

## 訓練スクリプト

### 2つの訓練モード

| モード | スクリプト | 設定ファイル | MST制約 |
|--------|----------|------------|---------|
| MST制約あり | `train_mst.py` | `configs/tree_2D_use_mst_only1.yaml` | 有効 |
| MST制約なし | `train_unmst.py` | `configs/tree_2D_use_unmst_only1.yaml` | 無効 |

### 実行コマンド

```bash
# MST制約付き訓練 (推奨)
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  train_mst.py \
  --config configs/tree_2D_use_mst_only1.yaml \
  --cuda_visible_device 0 1 2 3 4 5 6 7

# チェックポイントから再開
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  train_mst.py \
  --config configs/tree_2D_use_mst_only1.yaml \
  --cuda_visible_device 0 1 2 3 4 5 6 7 \
  --resume trained_weights/check/checkpoint_81_epoch.pkl
```

## ハイパーパラメータ

### モデル設定

```yaml
MODEL:
  NUM_CLASSES: 2

  ENCODER:
    HIDDEN_DIM: 128           # Transformer隠れ層次元
    NUM_FEATURE_LEVELS: 4     # マルチスケール特徴レベル数
    BACKBONE: resnet50        # バックボーン

  DECODER:
    HIDDEN_DIM: 128
    NHEADS: 8                 # Attention head数
    ENC_LAYERS: 4             # Encoderレイヤー数
    DEC_LAYERS: 4             # Decoderレイヤー数
    DIM_FEEDFORWARD: 128      # FFN次元
    DROPOUT: 0.15

    OBJ_TOKEN: 600            # 最大ノード数
    RLN_TOKEN: 1              # Relation token数
    RLN_ATTN: True            # Relation attention
```

### 訓練設定

```yaml
DATA:
  BATCH_SIZE: 8              # バッチサイズ
  IMG_SIZE: [512, 512]       # 入力画像サイズ
  NUM_WORKERS: 4             # データローダースレッド数
  SEED: 3407                 # 乱数シード

TRAIN:
  EPOCHS: 1000               # 総エポック数
  LR: 1e-4                   # 学習率
  LR_BACKBONE: 3e-5          # Backbone学習率
  WEIGHT_DECAY: 1e-4         # 重み減衰
  LR_DROP: 100               # 学習率減衰エポック
  CLIP_MAX_NORM: 0.1         # 勾配クリッピング

  VAL_INTERVAL: 1            # 検証間隔
  SAVE_VAL: True             # 検証結果を保存
```

### 損失の重み

```yaml
TRAIN:
  LOSSES: ['boxes', 'class', 'cards', 'nodes', 'edges']
  W_BBOX: 2.0
  W_CLASS: 3.0
  W_CARD: 1.0
  W_NODE: 5.0
  W_EDGE: 4.0
```

## オプティマイザー

### AdamW

**実装**: `train_mst.py:1000-1020`

```python
# パラメータグループ
param_dicts = [
    {
        "params": [p for n, p in model.named_parameters()
                   if "encoder" not in n and p.requires_grad]
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if "encoder" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]

# AdamW
optimizer = torch.optim.AdamW(
    param_dicts,
    lr=args.lr,
    weight_decay=args.weight_decay
)
```

**特徴**:
- **2つの学習率**: Backbone (3e-5) とその他 (1e-4)
- **AdamW**: L2正則化のある Adam
- **Weight Decay**: 1e-4

### 学習率スケジューラー

```python
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.lr_drop  # 100エポックごと
)
```

**スケジュール**:
- 初期: 1e-4
- 100エポック後: 1e-5
- 200エポック後: 1e-6
- ...

## 訓練ループ

### エポックごとの処理

**実装**: `epoch.py`

```python
for epoch in range(start_epoch, max_epoch):
    # 1. 訓練
    train_stats = train_one_epoch(
        model, criterion, data_loader_train,
        optimizer, device, epoch, args
    )

    # 2. 学習率更新
    lr_scheduler.step()

    # 3. チェックポイント保存
    if epoch % args.checkpoint_interval == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        torch.save(checkpoint, f'checkpoint_{epoch}_epoch.pkl')

    # 4. 検証
    if epoch % args.val_interval == 0:
        val_stats = validate(
            model, criterion, data_loader_val,
            device, epoch, args
        )
```

### 1エポックの訓練

**実装**: `epoch.py:train_one_epoch()`

```python
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, args):
    model.train()
    criterion.train()

    for samples, targets in data_loader:
        # Forward
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # Loss計算
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        # Backward
        optimizer.zero_grad()
        losses.backward()

        # 勾配クリッピング
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)

        optimizer.step()

    return train_stats
```

## MST制約の適用タイミング

### Forward Pass中

**場所**: `losses_only.py:loss_edges_mst_new()`

```python
# 訓練中、エッジ損失の計算時にMST制約が適用される
if args.use_mst_train:
    loss_edges = criterion.loss_edges_mst_new(
        h, target_nodes, target_edges, indices, epoch, max_epoch, last_epoch
    )
else:
    loss_edges = criterion.loss_edges(
        h, target_nodes, target_edges, indices
    )
```

**処理の流れ**:
1. エッジ確率を予測
2. コスト行列を構築
3. MST を計算 (scipy)
4. MSTに含まれないエッジのラベルを調整
5. 調整後の確率で損失を計算

## データ拡張

### 訓練時の拡張

**実装**: `train_mst.py:LoadCNNDataset`

```python
# 1. Brightness adjustment (20%)
if random.random() < 0.2:
    image = adjust_gamma(image, gamma=random.uniform(0.7, 1.3))

# 2. Gaussian noise (10%)
if random.random() < 0.1:
    image = gasuss_noise(image, mu=0.0, sigma=0.1)

# 3. Horizontal flip (50%)
if self.is_train and random.random() < 0.5:
    image = TF.hflip(image)
    points[:, 0] = 1 - points[:, 0]  # x座標を反転

# 4. Rotation (-15° to +15°) (if enabled)
if self.is_rotate and self.is_train:
    angle = random.uniform(-15, 15)
    image, points, edges = rotate_with_graph_validation(
        image, points, edges, angle
    )
```

### 検証時

```python
# 拡張なし、画像のみ正規化
image = TF.normalize(image, mean=[0.5], std=[0.5])
```

## 分散訓練

### DataParallel vs DistributedDataParallel

**使用**: `DistributedDataParallel` (DDP)

```python
# 初期化
torch.distributed.init_process_group(backend='nccl')

# モデルをDDPでラップ
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.local_rank],
    find_unused_parameters=True
)

# データローダー
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, sampler=sampler, ...)
```

**利点**:
- 複数GPUで効率的に訓練
- 各GPUが独立してforward/backwardを実行
- 勾配を自動で同期

## チェックポイントの管理

### 保存内容

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_scheduler': lr_scheduler.state_dict(),
    'epoch': epoch,
    'args': args,
}
```

### 読み込み

```python
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
```

## 訓練のモニタリング

### ログ出力

```python
# Epoch統計
log_stats = {
    'epoch': epoch,
    'train_loss': train_loss,
    'train_loss_class': train_loss_class,
    'train_loss_node': train_loss_node,
    'train_loss_edge': train_loss_edge,
    'train_loss_bbox': train_loss_bbox,
    'train_loss_card': train_loss_card,
    'lr': optimizer.param_groups[0]['lr'],
}

# JSON形式で保存
with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
    f.write(json.dumps(log_stats) + '\n')
```

### TensorBoard (未実装)

現在のコードには TensorBoard の統合はありませんが、追加可能:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=args.output_dir)
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/class', train_loss_class, epoch)
# ...
```

## メモリ管理

### 勾配累積 (未実装、必要に応じて)

```python
accumulation_steps = 4

for i, (samples, targets) in enumerate(data_loader):
    outputs = model(samples)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision (未実装、推奨)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(samples)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 訓練のベストプラクティス

### 1. ウォームアップ (推奨だが未実装)

```python
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
```

### 2. Early Stopping

```python
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(epochs):
    val_loss = validate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        save_checkpoint(...)
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break
```

### 3. 学習率ファインダー

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(data_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # 最適な学習率を視覚的に確認
```

## トラブルシューティング

### 損失がNaNになる

**原因**:
- 学習率が高すぎる
- 勾配爆発
- 数値不安定性

**解決策**:
```python
# 勾配クリッピングを強化
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

# NaNチェック
if torch.isnan(loss):
    print("NaN detected, skipping batch")
    continue

# 対数の安定化
relation_pred_softmax[relation_pred_softmax < 1e-24] = 1e-24
```

### メモリ不足

**解決策**:
1. バッチサイズを減らす
2. 勾配累積を使用
3. Mixed Precision訓練
4. OBJ_TOKENを減らす

### MST計算が遅い

**解決策**:
```python
# scipyの代わりにCUDA実装を使用 (要実装)
# または、エポックごとにMST制約を適用しない
if epoch % 5 == 0:  # 5エポックごと
    use_mst = True
else:
    use_mst = False
```

## 訓練時間の目安

| 設定 | GPU | バッチサイズ | 1エポック時間 | 1000エポック |
|-----|-----|------------|------------|-----------|
| 標準 | 8xV100 | 8 | ~5分 | ~3.5日 |
| MST制約 | 8xV100 | 8 | ~8分 | ~5.5日 |
| 小規模 | 1xV100 | 2 | ~15分 | ~10日 |

## まとめ

TreeFormerの訓練戦略:
1. **AdamW + StepLR**: 安定した最適化
2. **MST制約**: 訓練中に木構造を保証
3. **分散訓練**: 複数GPUで効率化
4. **データ拡張**: ロバスト性向上
5. **勾配クリッピング**: 数値安定性
