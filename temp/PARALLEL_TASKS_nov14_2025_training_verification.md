# TreeFormer Training Verification - Parallel Execution Tasks
## Guyot Dataset学習検証：並列実行可能タスク構成

**作成日**: 2025-11-14
**ブランチ**: `claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS`
**目的**: TreeFormer学習検証作業を並列実行可能なSubagent Task形式で構成

---

## 📋 概要

### 並列実行戦略
- **Stage 0-1**: 完全並列実行可能（相互依存なし）
- **Stage 2**: 一部並列実行可能（Config作成とpackageインストールは並列可）
- **Stage 3-5**: 順次実行（前Stage完了が必須）

### Task命名規則
- **A系**: Stage 0事前調査（完全並列）
- **B系**: Stage 1テストスクリプト作成（完全並列）
- **C系**: Stage 2 Config・依存関係セットアップ（一部並列）
- **D系**: Stage 3 Dataset Loader検証（順次）
- **E系**: Stage 4 Training実行（順次）
- **F系**: Stage 5レポート作成（順次）

---

## Stage 0: 事前調査フェーズ（完全並列実行可能）

### 🔵 Task A1: Dataset実ファイル確認

**依存関係**: なし（独立実行可能）
**実行モデル**: haiku（軽量タスク）
**想定時間**: 2-3分

#### 入力（前提条件）
- `data/guyot_200_20_resized/`ディレクトリが存在
- Bashコマンド実行権限

#### 実行内容
1. Train/Validationセットの存在確認
   ```bash
   ls -la data/guyot_200_20_resized/01-TrainAndValidationSet/ | head -20
   ```

2. Train画像数カウント
   ```bash
   find data/guyot_200_20_resized/01-TrainAndValidationSet/ -name "*.jpeg" | wc -l
   ```
   - **期待値**: 200

3. Trainアノテーション数カウント
   ```bash
   find data/guyot_200_20_resized/01-TrainAndValidationSet/ -name "*_annotation.json" | wc -l
   ```
   - **期待値**: 200

4. Test画像数カウント
   ```bash
   find data/guyot_200_20_resized/02-IndependentTestSet/ -name "*.jpeg" | wc -l
   ```
   - **期待値**: 20

5. サンプル画像サイズ確認
   ```bash
   file data/guyot_200_20_resized/01-TrainAndValidationSet/Set*.jpeg | head -3
   ```
   - **期待値**: 512x512が表示される

#### 出力（成果物）
- `/tmp/task_a1_dataset_check.json`
  ```json
  {
    "train_images": 200,
    "train_annotations": 200,
    "test_images": 20,
    "sample_image_size": "512x512",
    "status": "success"
  }
  ```

#### 検証方法
```bash
cat /tmp/task_a1_dataset_check.json | jq '.status'
# 期待出力: "success"
```

#### エラー処理
- **画像数不一致**: `status: "error"`, `error_message: "Expected 200 train images, found X"`
- **ディレクトリ不存在**: `status: "error"`, `error_message: "Dataset directory not found"`

---

### 🔵 Task A2: 既存Config全パラメータ解析

**依存関係**: なし（独立実行可能）
**実行モデル**: haiku
**想定時間**: 2-3分

#### 入力（前提条件）
- `configs/tree_2D_use_mst_only1.yaml`が存在
- Python + PyYAML利用可能

#### 実行内容
1. Config読み込み
   ```bash
   cat configs/tree_2D_use_mst_only1.yaml
   ```

2. DATA section抽出
   ```bash
   grep -A 10 "^DATA:" configs/tree_2D_use_mst_only1.yaml
   ```

3. MODEL section抽出
   ```bash
   grep -A 30 "^MODEL:" configs/tree_2D_use_mst_only1.yaml
   ```

4. TRAIN section抽出
   ```bash
   grep -A 15 "^TRAIN:" configs/tree_2D_use_mst_only1.yaml
   ```

5. Pythonで構造化データ抽出
   ```python
   import yaml
   with open('configs/tree_2D_use_mst_only1.yaml', 'r') as f:
       config = yaml.safe_load(f)

   result = {
       'data_path': config['DATA']['DATA_PATH'],
       'dataset': config['DATA']['DATASET'],
       'img_size': config['DATA']['IMG_SIZE'],
       'batch_size': config['DATA']['BATCH_SIZE'],
       'epochs': config['TRAIN']['EPOCHS'],
       'learning_rate': config['TRAIN']['LR'],
       'obj_token': config['MODEL']['DECODER']['OBJ_TOKEN']
   }
   ```

#### 出力（成果物）
- `/tmp/task_a2_config_analysis.json`
  ```json
  {
    "data_path": "./data/toulouse-road-network",
    "dataset": "toulouse-road-network-2D",
    "img_size": [512, 512],
    "batch_size": 8,
    "epochs": 1000,
    "learning_rate": 0.0001,
    "obj_token": 600,
    "guyot_required_changes": {
      "data_path": "./data/guyot_200_20_resized",
      "dataset": "guyot-2D",
      "epochs": 2
    },
    "status": "success"
  }
  ```

#### 検証方法
```bash
cat /tmp/task_a2_config_analysis.json | jq '.status'
# 期待出力: "success"
```

---

### 🔵 Task A3: train_mst.py コード解析

**依存関係**: なし（独立実行可能）
**実行モデル**: sonnet（複雑な解析）
**想定時間**: 5-7分

#### 入力（前提条件）
- `train_mst.py`が存在
- Grep/Read tool利用可能

#### 実行内容
1. Import文一覧抽出
   ```bash
   grep "^import\|^from.*import" train_mst.py | sort -u > /tmp/train_mst_imports.txt
   ```

2. Dataset初期化箇所特定
   ```bash
   grep -n "dataset.*=" train_mst.py | grep -i "train\|test\|val" > /tmp/train_mst_dataset_init.txt
   ```

3. DataLoader作成箇所特定
   ```bash
   grep -n "DataLoader" train_mst.py > /tmp/train_mst_dataloader.txt
   ```

4. Argument parser確認
   ```bash
   grep -A 20 "ArgumentParser\|argparse" train_mst.py > /tmp/train_mst_argparse.txt
   ```

5. Dataset class特定
   ```bash
   # 該当行周辺を読み込み、実際のdataset class名を特定
   # 例: ToulouseDataset, GenericTreeDataset等
   ```

#### 出力（成果物）
- `/tmp/task_a3_train_mst_analysis.json`
  ```json
  {
    "dataset_class": "GenericTreeDataset",
    "dataset_init_line": 520,
    "dataloader_line": 580,
    "required_args": ["--config"],
    "optional_args": ["--resume", "--device", "--use_mst_train"],
    "dataset_module": "datasets.tree_dataset",
    "collate_fn_used": true,
    "status": "success"
  }
  ```

#### 検証方法
```bash
cat /tmp/task_a3_train_mst_analysis.json | jq '.dataset_class'
# 期待出力: dataset class名が特定されている
```

---

### 🔵 Task A4: 依存関係確認

**依存関係**: なし（独立実行可能）
**実行モデル**: haiku
**想定時間**: 3-4分

#### 入力（前提条件）
- `.venv/`仮想環境が存在
- `pip list`実行可能

#### 実行内容
1. requirements.txt/pyproject.toml確認
   ```bash
   ls -la requirements.txt pyproject.toml 2>/dev/null || echo "No dependency file"
   ```

2. 現在インストール済みpackage一覧
   ```bash
   source .venv/bin/activate
   pip list > /tmp/current_packages.txt
   ```

3. 主要package確認
   ```bash
   pip list | grep -E "torch|numpy|scipy|networkx|opencv|Pillow|PyYAML|tqdm|matplotlib"
   ```

4. train_mst.pyから必要packageリスト作成
   ```bash
   grep "^import\|^from.*import" train_mst.py | awk '{print $2}' | cut -d'.' -f1 | sort -u > /tmp/required_packages_raw.txt
   ```

5. 不足package特定
   ```python
   # package名の正規化マップ
   package_map = {
       'cv2': 'opencv-python',
       'PIL': 'Pillow',
       'yaml': 'PyYAML',
       'skimage': 'scikit-image'
   }
   ```

#### 出力（成果物）
- `/tmp/task_a4_dependency_check.json`
  ```json
  {
    "installed_packages": {
      "torch": "2.0.1",
      "numpy": "1.24.3",
      "networkx": "3.1"
    },
    "missing_packages": [
      "opencv-python",
      "scikit-image"
    ],
    "dependency_file_exists": false,
    "status": "success"
  }
  ```

#### 検証方法
```bash
cat /tmp/task_a4_dependency_check.json | jq '.missing_packages | length'
# 不足packageの数を確認
```

---

## Stage 1: テストスクリプト作成（完全並列実行可能）

### 🔵 Task B1: 画像読み込みテストスクリプト作成

**依存関係**: Task A1完了（dataset存在確認済み）
**実行モデル**: haiku
**想定時間**: 2分

#### 実行内容
```python
cat > /tmp/test_image_load.py << 'EOF'
import sys
from PIL import Image
import json

image_path = sys.argv[1]
annotation_path = sys.argv[2]

# 画像読み込み
img = Image.open(image_path)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")

# アノテーション読み込み
with open(annotation_path, 'r') as f:
    anno = json.load(f)
print(f"Annotation keys: {list(anno.keys())}")
if 'VineFeature' in anno:
    print(f"Number of VineFeatures: {len(anno['VineFeature'])}")
EOF
```

#### 出力（成果物）
- `/tmp/test_image_load.py`

#### 検証方法
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_image_load.py \
  data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001.jpeg \
  data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001_annotation.json
# 期待: Image size: (512, 512)
```

---

### 🔵 Task B2: Config読み込みテストスクリプト作成

**依存関係**: Task A2完了
**実行モデル**: haiku
**想定時間**: 2分

#### 実行内容
```python
cat > /tmp/test_config_load.py << 'EOF'
import yaml
import sys

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("=== DATA Section ===")
for key, value in config.get('DATA', {}).items():
    print(f"  {key}: {value}")

print("\n=== MODEL.DECODER Section ===")
decoder = config.get('MODEL', {}).get('DECODER', {})
for key in ['OBJ_TOKEN', 'RLN_TOKEN', 'HIDDEN_DIM', 'ENC_LAYERS', 'DEC_LAYERS']:
    print(f"  {key}: {decoder.get(key)}")

print("\n=== TRAIN Section ===")
for key, value in config.get('TRAIN', {}).items():
    print(f"  {key}: {value}")
EOF
```

#### 出力（成果物）
- `/tmp/test_config_load.py`

---

### 🔵 Task B3: Dataset importテストスクリプト作成

**依存関係**: Task A3完了（dataset class特定済み）
**実行モデル**: haiku
**想定時間**: 3分

#### 実行内容
```python
cat > /tmp/test_dataset_import.py << 'EOF'
import sys
sys.path.insert(0, '/home/user/TreeFormer')

# Task A3で特定したdataset classをimport
# 例:
dataset_candidates = [
    ('datasets.guyot_dataset', 'GuyotDataset'),
    ('datasets.tree_dataset', 'TreeDataset'),
    ('datasets.generic_dataset', 'GenericDataset')
]

found_datasets = []
for module, cls in dataset_candidates:
    try:
        mod = __import__(module, fromlist=[cls])
        dataset_cls = getattr(mod, cls)
        print(f"✓ {cls} found in {module}")
        found_datasets.append((module, cls))
    except (ImportError, AttributeError) as e:
        print(f"✗ {cls} not found in {module}: {e}")

if found_datasets:
    print(f"\n✓ Total datasets found: {len(found_datasets)}")
else:
    print("\n✗ No dataset classes found")
    sys.exit(1)
EOF
```

#### 出力（成果物）
- `/tmp/test_dataset_import.py`
- Task A3の結果を反映したdataset候補リスト

---

### 🔵 Task B4: 依存packageインポートテストスクリプト作成

**依存関係**: なし
**実行モデル**: haiku
**想定時間**: 2分

#### 実行内容
```python
cat > /tmp/test_imports.py << 'EOF'
import sys

packages = [
    'torch',
    'torchvision',
    'numpy',
    'scipy',
    'networkx',
    'cv2',
    'PIL',
    'yaml',
    'tqdm',
    'matplotlib'
]

failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError as e:
        print(f"✗ {pkg}: {e}")
        failed.append(pkg)

if failed:
    print(f"\n✗ {len(failed)} packages failed to import")
    sys.exit(1)
else:
    print(f"\n✓ All {len(packages)} packages imported successfully")
EOF
```

#### 出力（成果物）
- `/tmp/test_imports.py`

---

## Stage 2: Config・依存関係セットアップ（一部並列可）

### 🔶 Task C1: Guyot用Config作成

**依存関係**: Task A1, A2完了
**実行モデル**: sonnet
**想定時間**: 3-4分

#### 実行内容
1. ベースConfigコピー
   ```bash
   cp configs/tree_2D_use_mst_only1.yaml configs/tree_2D_guyot_test.yaml
   ```

2. 必要箇所を編集（Edit toolを使用）
   - `DATA_PATH: './data/guyot_200_20_resized'`
   - `DATASET: 'guyot-2D'`
   - `EPOCHS: 2`
   - `exp_name: 'test_guyot_baseline_nov14_2025'`

3. 差分確認
   ```bash
   diff -u configs/tree_2D_use_mst_only1.yaml configs/tree_2D_guyot_test.yaml
   ```

#### 出力（成果物）
- `configs/tree_2D_guyot_test.yaml`

#### 検証方法
```bash
python /tmp/test_config_load.py configs/tree_2D_guyot_test.yaml | grep "DATA_PATH"
# 期待: guyot_200_20_resizedが含まれる
```

---

### 🔶 Task C2: Config差分検証

**依存関係**: Task C1完了
**実行モデル**: haiku
**想定時間**: 2分

#### 実行内容
```python
cat > /tmp/test_config_diff.py << 'EOF'
import yaml

with open('configs/tree_2D_use_mst_only1.yaml', 'r') as f:
    config_old = yaml.safe_load(f)
with open('configs/tree_2D_guyot_test.yaml', 'r') as f:
    config_new = yaml.safe_load(f)

# 変更点チェック
assert config_new['DATA']['DATA_PATH'] == './data/guyot_200_20_resized', "DATA_PATH not updated"
assert config_new['DATA']['DATASET'] == 'guyot-2D', "DATASET not updated"
assert config_new['TRAIN']['EPOCHS'] == 2, "EPOCHS not updated"
assert config_new['log']['exp_name'] == 'test_guyot_baseline_nov14_2025', "exp_name not updated"

# 変更してはいけない項目チェック
assert config_new['DATA']['IMG_SIZE'] == config_old['DATA']['IMG_SIZE'], "IMG_SIZE should not change"
assert config_new['DATA']['BATCH_SIZE'] == config_old['DATA']['BATCH_SIZE'], "BATCH_SIZE should not change"
assert config_new['MODEL'] == config_old['MODEL'], "MODEL config should not change"

print("✓ All config changes validated successfully")
EOF

python /tmp/test_config_diff.py
```

#### 出力（成果物）
- Validation通過確認

---

### 🔶 Task C3: 依存package不足分インストール

**依存関係**: Task A4完了
**実行モデル**: haiku
**想定時間**: 5-10分（インストール時間による）
**並列実行**: Task C1と並列実行可能

#### 実行内容
1. 不足package特定（Task A4結果から）
   ```bash
   cat /tmp/task_a4_dependency_check.json | jq -r '.missing_packages[]'
   ```

2. 一括インストール
   ```bash
   cd /home/user/TreeFormer
   source .venv/bin/activate

   # /tmp/task_a4_dependency_check.jsonから不足packageを読み込み
   pip install $(cat /tmp/task_a4_dependency_check.json | jq -r '.missing_packages[]')
   ```

3. インストール確認
   ```bash
   python /tmp/test_imports.py
   ```

#### 出力（成果物）
- インストール完了確認
- `/tmp/test_imports.py`実行結果: all ✓

---

## Stage 3: Dataset Loader検証（順次実行）

### 🟡 Task D1: Dataset読み込みテスト

**依存関係**: Task A3, B3, C1, C3完了
**実行モデル**: sonnet
**想定時間**: 5-7分

#### 実行内容
1. Task A3で特定したdataset classを使用してtest script作成
   ```python
   cat > /tmp/test_dataset_load.py << 'EOF'
   import sys
   sys.path.insert(0, '/home/user/TreeFormer')
   import yaml

   # Task A3の結果に基づいてimport
   from datasets.tree_dataset import TreeDataset  # 例

   with open('configs/tree_2D_guyot_test.yaml', 'r') as f:
       config = yaml.safe_load(f)

   # Dataset初期化（実際のtrain_mst.pyのコードを参照）
   dataset = TreeDataset(config, split='train')

   print(f"✓ Dataset loaded successfully")
   print(f"  Dataset size: {len(dataset)}")
   print(f"  Dataset path: {config['DATA']['DATA_PATH']}")

   # 1サンプル取得テスト
   sample = dataset[0]
   print(f"  Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'tensor'}")
   EOF
   ```

2. 実行
   ```bash
   cd /home/user/TreeFormer
   source .venv/bin/activate
   python /tmp/test_dataset_load.py
   ```

#### 出力（成果物）
- `/tmp/test_dataset_load.py`
- 実行結果: Dataset size 200確認

#### エラー処理
- **Dataset class不明**: train_mst.pyから該当コードを直接抽出して使用
- **FileNotFoundError**: パス確認、絶対パス使用を検討

---

### 🟡 Task D2: DataLoader batch取得テスト

**依存関係**: Task D1完了
**実行モデル**: sonnet
**想定時間**: 4-5分

#### 実行内容
1. DataLoader作成test script
   ```python
   cat > /tmp/test_dataloader.py << 'EOF'
   import sys
   sys.path.insert(0, '/home/user/TreeFormer')
   import yaml
   import torch
   from torch.utils.data import DataLoader

   # Task D1で確認したdataset
   from datasets.tree_dataset import TreeDataset

   with open('configs/tree_2D_guyot_test.yaml', 'r') as f:
       config = yaml.safe_load(f)

   dataset = TreeDataset(config, split='train')

   # DataLoader作成（collate_fnの有無はtrain_mst.py参照）
   dataloader = DataLoader(
       dataset,
       batch_size=config['DATA']['BATCH_SIZE'],
       num_workers=config['DATA']['NUM_WORKERS'],
       shuffle=True
   )

   # 1 batch取得
   for batch in dataloader:
       print(f"✓ Batch retrieved successfully")
       print(f"  Batch type: {type(batch)}")
       if isinstance(batch, dict):
           for key, value in batch.items():
               if isinstance(value, torch.Tensor):
                   print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
       break
   EOF
   ```

2. 実行
   ```bash
   cd /home/user/TreeFormer
   source .venv/bin/activate
   python /tmp/test_dataloader.py
   ```

#### 出力（成果物）
- `/tmp/test_dataloader.py`
- Batch shape確認（例: images [8, 3, 512, 512]）

---

## Stage 4: Training実行（順次実行）

### 🟢 Task E1: DRY RUN (0 epoch)

**依存関係**: Task C1, C3, D1, D2完了
**実行モデル**: sonnet
**想定時間**: 3-5分

#### 実行内容
1. DRY RUN用Config作成
   ```bash
   cp configs/tree_2D_guyot_test.yaml configs/tree_2D_guyot_dry_run.yaml
   sed -i 's/EPOCHS: 2/EPOCHS: 0/' configs/tree_2D_guyot_dry_run.yaml
   ```

2. DRY RUN実行
   ```bash
   cd /home/user/TreeFormer
   source .venv/bin/activate
   python train_mst.py --config configs/tree_2D_guyot_dry_run.yaml --device cuda 2>&1 | tee /tmp/dry_run_log.txt
   ```

#### 出力（成果物）
- `/tmp/dry_run_log.txt`
- エラーなく終了確認

#### エラー処理
- **CUDA OOM**: BATCH_SIZE削減（8→4→2）
- **ModuleNotFoundError**: Phase 3に戻りpackage追加

---

### 🟢 Task E2: 1 Epoch Training実行

**依存関係**: Task E1完了（DRY RUN成功）
**実行モデル**: sonnet
**想定時間**: 15-30分（データサイズとGPU性能による）

#### 実行内容
1. 1 Epoch config作成
   ```bash
   cp configs/tree_2D_guyot_test.yaml configs/tree_2D_guyot_1epoch.yaml
   sed -i 's/EPOCHS: 2/EPOCHS: 1/' configs/tree_2D_guyot_1epoch.yaml
   sed -i 's/test_guyot_baseline_nov14_2025/test_guyot_1epoch_nov14_2025/' configs/tree_2D_guyot_1epoch.yaml
   ```

2. 1 Epoch実行
   ```bash
   cd /home/user/TreeFormer
   source .venv/bin/activate
   time python train_mst.py --config configs/tree_2D_guyot_1epoch.yaml --device cuda 2>&1 | tee /tmp/1epoch_train_log.txt
   ```

#### 出力（成果物）
- `trained_weights/test_guyot_1epoch_nov14_2025/*.pth`
- `trained_weights/test_guyot_1epoch_nov14_2025/train.log`
- `/tmp/1epoch_train_log.txt`

---

### 🟢 Task E3: Checkpoint検証

**依存関係**: Task E2完了
**実行モデル**: haiku
**想定時間**: 2-3分

#### 実行内容
```python
cat > /tmp/test_checkpoint.py << 'EOF'
import torch
import sys
import glob

checkpoint_dir = 'trained_weights/test_guyot_1epoch_nov14_2025/'
checkpoint_files = glob.glob(checkpoint_dir + '*.pth')

if not checkpoint_files:
    print("✗ No checkpoint found")
    sys.exit(1)

checkpoint_path = checkpoint_files[0]
print(f"Loading checkpoint: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Keys: {list(checkpoint.keys())}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'model_state_dict' in checkpoint:
        print(f"  Model state dict size: {len(checkpoint['model_state_dict'])}")
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    sys.exit(1)
EOF

cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_checkpoint.py
```

#### 出力（成果物）
- Checkpoint読み込み成功確認

---

### 🟢 Task E4: Loss解析と可視化

**依存関係**: Task E2完了
**実行モデル**: haiku
**想定時間**: 3-4分

#### 実行内容
```python
cat > /tmp/plot_loss.py << 'EOF'
import re
import matplotlib.pyplot as plt

log_file = 'trained_weights/test_guyot_1epoch_nov14_2025/train.log'

losses = []
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(r'loss[=:\s]+([\d.]+)', line, re.IGNORECASE)
        if match:
            losses.append(float(match.group(1)))

if losses:
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss (1 Epoch)')
    plt.grid(True)
    plt.savefig('/tmp/loss_plot.png')
    print(f"✓ Loss plot saved to /tmp/loss_plot.png")
    print(f"  Total iterations: {len(losses)}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")

    # JSON出力
    import json
    with open('/tmp/task_e4_loss_analysis.json', 'w') as f:
        json.dump({
            'total_iterations': len(losses),
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'converged': losses[-1] < losses[0],
            'status': 'success'
        }, f, indent=2)
else:
    print("✗ No loss values found in log")
EOF

cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/plot_loss.py
```

#### 出力（成果物）
- `/tmp/loss_plot.png`
- `/tmp/task_e4_loss_analysis.json`

---

## Stage 5: レポート作成（順次実行）

### 🟣 Task F1: Training検証レポート作成

**依存関係**: Task E2, E3, E4完了
**実行モデル**: sonnet
**想定時間**: 5-7分

#### 実行内容
1. 全Task結果を集約
   ```bash
   # Task結果JSONファイルを収集
   cat /tmp/task_a1_dataset_check.json
   cat /tmp/task_a2_config_analysis.json
   cat /tmp/task_a4_dependency_check.json
   cat /tmp/task_e4_loss_analysis.json
   ```

2. レポート作成
   ```markdown
   # TreeFormer Guyot Dataset Training Verification Report
   **Date**: 2025-11-14
   **Config**: configs/tree_2D_guyot_1epoch.yaml
   **Branch**: claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS

   ## Summary
   - ✓ Dataset loading: 200 train, 20 test images
   - ✓ 1 Epoch training completion: Yes
   - ✓ Checkpoint generation: Yes
   - ✓ Loss convergence: Initial X.XX → Final Y.YY

   ## Parallel Execution Results
   - Stage 0 (4 tasks): X min
   - Stage 1 (4 tasks): X min
   - Stage 2 (3 tasks): X min
   - Stage 3 (2 tasks): X min
   - Stage 4 (4 tasks): X min

   ## Training Results
   - Epochs completed: 1
   - Total iterations: <from task_e4>
   - Initial loss: <from task_e4>
   - Final loss: <from task_e4>
   - Training time: <from log>

   ## Files Generated
   - Checkpoint: trained_weights/test_guyot_1epoch_nov14_2025/*.pth
   - Log: trained_weights/test_guyot_1epoch_nov14_2025/train.log
   - Loss plot: /tmp/loss_plot.png

   ## Issues Encountered
   <各Task結果からerrorを抽出>

   ## Next Steps
   1. Full training (100+ epochs) with validation
   2. Hyperparameter tuning
   3. Evaluation on test set
   4. Comparison with baseline

   ## Conclusion
   Guyot datasetでのTreeFormer学習が正常に動作することを確認した。
   並列実行により作業時間をX%削減できた。
   ```

#### 出力（成果物）
- `temp/PARALLEL_TRAINING_VERIFICATION_REPORT_nov14_2025.md`

---

## 📊 実行指示書（Subagent用）

### Stage 0実行（完全並列）
```bash
# 以下4タスクを並列実行
parallel_execute {
  task_a1: "Dataset実ファイル確認",
  task_a2: "既存Config解析",
  task_a3: "train_mst.py解析",
  task_a4: "依存関係確認"
}
```

### Stage 1実行（完全並列）
```bash
# Stage 0完了後、以下4タスクを並列実行
wait_for: [task_a1, task_a2, task_a3, task_a4]
parallel_execute {
  task_b1: "画像読み込みテストスクリプト作成",
  task_b2: "Config読み込みテストスクリプト作成",
  task_b3: "Dataset importテストスクリプト作成",
  task_b4: "依存packageインポートテストスクリプト作成"
}
```

### Stage 2実行（一部並列）
```bash
# Task C1とC3は並列実行可能
wait_for: [task_a1, task_a2, task_b1, task_b2]
parallel_execute {
  task_c1: "Guyot用Config作成",
  task_c3: "依存package不足分インストール"  # Task A4結果に依存
}

# Task C2はC1完了後に実行
wait_for: [task_c1]
execute: task_c2
```

### Stage 3実行（順次）
```bash
wait_for: [task_a3, task_b3, task_c1, task_c3]
execute: task_d1
wait_for: [task_d1]
execute: task_d2
```

### Stage 4実行（順次）
```bash
wait_for: [task_c1, task_c3, task_d1, task_d2]
execute: task_e1
wait_for: [task_e1]
execute: task_e2
wait_for: [task_e2]
parallel_execute {
  task_e3: "Checkpoint検証",
  task_e4: "Loss解析と可視化"
}
```

### Stage 5実行（順次）
```bash
wait_for: [task_e2, task_e3, task_e4]
execute: task_f1
```

---

## 🔄 Task依存関係グラフ

```
Stage 0 (並列)
├─ A1: Dataset確認 ────────┐
├─ A2: Config解析 ─────────┤
├─ A3: train_mst解析 ──────┤
└─ A4: 依存関係確認 ───────┤
                           ↓
Stage 1 (並列)
├─ B1: 画像テストスクリプト ┐
├─ B2: Configテストスクリプト┤
├─ B3: Datasetテストスクリプト┤
└─ B4: 依存テストスクリプト ──┤
                           ↓
Stage 2 (一部並列)
├─ C1: Config作成 ─────────┐
├─ C3: package install ────┤ (並列)
└─ C2: Config検証 ─────────┘ (C1後)
                           ↓
Stage 3 (順次)
├─ D1: Dataset読み込み ────┐
└─ D2: DataLoader batch ───┘
                           ↓
Stage 4 (順次→最後並列)
├─ E1: DRY RUN ────────────┐
├─ E2: 1 Epoch Training ───┤
├─ E3: Checkpoint検証 ─────┤ (E2後、並列)
└─ E4: Loss解析 ───────────┘ (E2後、並列)
                           ↓
Stage 5 (順次)
└─ F1: レポート作成
```

---

## 📝 成果物一覧

### JSON結果ファイル
- `/tmp/task_a1_dataset_check.json`
- `/tmp/task_a2_config_analysis.json`
- `/tmp/task_a3_train_mst_analysis.json`
- `/tmp/task_a4_dependency_check.json`
- `/tmp/task_e4_loss_analysis.json`

### テストスクリプト
- `/tmp/test_image_load.py`
- `/tmp/test_config_load.py`
- `/tmp/test_dataset_import.py`
- `/tmp/test_imports.py`
- `/tmp/test_config_diff.py`
- `/tmp/test_dataset_load.py`
- `/tmp/test_dataloader.py`
- `/tmp/test_checkpoint.py`
- `/tmp/plot_loss.py`

### Configファイル
- `configs/tree_2D_guyot_test.yaml`
- `configs/tree_2D_guyot_dry_run.yaml`
- `configs/tree_2D_guyot_1epoch.yaml`

### Training成果物
- `trained_weights/test_guyot_1epoch_nov14_2025/*.pth`
- `trained_weights/test_guyot_1epoch_nov14_2025/train.log`
- `/tmp/loss_plot.png`

### ドキュメント
- `temp/PARALLEL_TRAINING_VERIFICATION_REPORT_nov14_2025.md`

---

## 🚀 実行開始コマンド

### Subagent並列実行（推奨）
```bash
# Stage 0: 4 tasks並列
claude_code task --parallel \
  --task-a1 "Dataset実ファイル確認" \
  --task-a2 "既存Config解析" \
  --task-a3 "train_mst.py解析" \
  --task-a4 "依存関係確認"

# Stage 1: 4 tasks並列
claude_code task --parallel \
  --task-b1 "画像読み込みテストスクリプト作成" \
  --task-b2 "Config読み込みテストスクリプト作成" \
  --task-b3 "Dataset importテストスクリプト作成" \
  --task-b4 "依存packageインポートテストスクリプト作成"

# Stage 2-5: 順次または一部並列
```

### 手動実行（Stage単位）
```bash
# Stage 0-1を手動で実行する場合、各TaskのBashコマンドを
# 異なるターミナルウィンドウで同時実行可能
```

---

**END OF PARALLEL TASKS DOCUMENT**
