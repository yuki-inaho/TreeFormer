# TreeFormer Training Verification Work Plan
## Guyot Datasetでの学習動作確認作業書

**作成日**: 2025-11-14
**ブランチ**: `claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS`
**目的**: `data/guyot_200_20_resized/`の実データを用いてTreeFormer学習が正常動作することを確認

---

## 📑 作業方針

### 開発原則
- **DRY (Don't Repeat Yourself)**: 重複コード・設定を避ける
- **KISS (Keep It Simple, Stupid)**: シンプルで理解しやすい実装
- **SOLID原則**: 特にSingle Responsibility, Dependency Inversion
- **t-wada TDD**: Test-Driven Development、テストファースト

### 実行ルール
1. **アトミック性**: 1ステップ = 1操作、分割不可
2. **順序性**: 依存関係を明示、順番厳守
3. **検証可能性**: 各ステップに成功条件を明記
4. **暗黙的フォールバック禁止**: エラー時は明示的に処理、推測しない
5. **完結性**: 各ステップは独立して完結可能

### 記号凡例
- 🖐 **操作**: コマンド実行、ファイル編集等の実作業
- 🔎 **確認**: 結果検証、ログ確認等
- 🧪 **テスト**: 自動テスト実行、TDD要素
- 🛠 **エラー時対処**: 問題発生時の具体的対応手順

---

## Phase 1: Dataset & Config 事前調査

### ✅ チェックリスト
- [ ] 1.1 Dataset実ファイル確認
- [ ] 1.2 既存Config全パラメータ解析
- [ ] 1.3 Guyot用Config作成
- [ ] 1.4 Config差分確認テスト

---

### Task 1.1: Dataset実ファイル確認

#### 🖐 Step 1.1.1: Train/Validationセット確認
```bash
ls -la data/guyot_200_20_resized/01-TrainAndValidationSet/ | head -20
```

**期待結果**:
- `Set*.jpeg`ファイルが存在
- `Set*_annotation.json`ファイルが存在
- 合計400ファイル程度 (200 images + 200 annotations)

#### 🔎 Step 1.1.2: 画像数カウント
```bash
find data/guyot_200_20_resized/01-TrainAndValidationSet/ -name "*.jpeg" | wc -l
```

**成功条件**: `200`が出力される

#### 🔎 Step 1.1.3: アノテーション数カウント
```bash
find data/guyot_200_20_resized/01-TrainAndValidationSet/ -name "*_annotation.json" | wc -l
```

**成功条件**: `200`が出力される

#### 🖐 Step 1.1.4: Testセット確認
```bash
ls -la data/guyot_200_20_resized/02-IndependentTestSet/ | head -20
```

**期待結果**:
- `Set*.jpeg`ファイルが存在
- `Set*_annotation.json`ファイルが存在
- 合計40ファイル程度 (20 images + 20 annotations)

#### 🔎 Step 1.1.5: Test画像数カウント
```bash
find data/guyot_200_20_resized/02-IndependentTestSet/ -name "*.jpeg" | wc -l
```

**成功条件**: `20`が出力される

#### 🖐 Step 1.1.6: サンプル画像サイズ確認
```bash
file data/guyot_200_20_resized/01-TrainAndValidationSet/Set*.jpeg | head -3
```

**期待結果**: 解像度が表示される（512x512を期待）

#### 🧪 Step 1.1.7: 画像読み込みテストスクリプト作成
```bash
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

#### 🧪 Step 1.1.8: 画像読み込みテスト実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_image_load.py \
  data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001.jpeg \
  data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001_annotation.json
```

**成功条件**:
- `Image size: (512, 512)`が出力
- `Image mode: RGB`が出力
- `Annotation keys`にVineFeature等が含まれる
- エラーなく完了

#### 🛠 Step 1.1.9: エラー時対処 - 画像が存在しない
**症状**: `FileNotFoundError: [Errno 2] No such file or directory`

**対処**:
```bash
# Dataset全体を確認
find data/ -type d -name "guyot*"
# 正しいパスを特定し、Step 1.1.1から再実行
```

#### 🛠 Step 1.1.10: エラー時対処 - 画像サイズ不一致
**症状**: 512x512以外のサイズが出力

**対処**:
```bash
# 全画像のサイズを確認
find data/guyot_200_20_resized/ -name "*.jpeg" -exec identify -format "%f: %wx%h\n" {} \; | sort -u
# サイズが混在している場合、tools/resize_guyot_dataset.pyの再実行を検討
```

---

### Task 1.2: 既存Config全パラメータ解析

#### 🖐 Step 1.2.1: Config読み込み
```bash
cat configs/tree_2D_use_mst_only1.yaml
```

#### 🔎 Step 1.2.2: Dataset関連パラメータ抽出
```bash
grep -A 10 "^DATA:" configs/tree_2D_use_mst_only1.yaml
```

**確認項目**:
- `DATA_PATH`: './data/toulouse-road-network' → Guyot用に変更必要
- `DATASET`: 'toulouse-road-network-2D' → 変更必要
- `IMG_SIZE`: [512, 512] → Guyotと一致、変更不要
- `BATCH_SIZE`: 8 → そのまま使用可能
- `NUM_WORKERS`: 4 → 環境に応じて調整可能

#### 🔎 Step 1.2.3: Model関連パラメータ抽出
```bash
grep -A 30 "^MODEL:" configs/tree_2D_use_mst_only1.yaml
```

**確認項目**:
- `ENCODER.TYPE`: deformable_transformer_backbone
- `ENCODER.HIDDEN_DIM`: 128
- `DECODER.TYPE`: deformable_transformer
- `DECODER.OBJ_TOKEN`: 600 → Guyot datasetのmax node数と比較必要

#### 🔎 Step 1.2.4: Training関連パラメータ抽出
```bash
grep -A 15 "^TRAIN:" configs/tree_2D_use_mst_only1.yaml
```

**確認項目**:
- `EPOCHS`: 1000 → テスト時は1-2に変更
- `LR`: 1e-4
- `SAVE_PATH`: "./trained_weights/"
- `LOSSES`: ['boxes', 'class', 'cards', 'nodes', 'edges']

#### 🧪 Step 1.2.5: Config読み込みテストスクリプト作成
```bash
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

#### 🧪 Step 1.2.6: Config読み込みテスト実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_config_load.py configs/tree_2D_use_mst_only1.yaml
```

**成功条件**: YAMLが正常にparse され、全セクションが表示される

---

### Task 1.3: Guyot用Config作成

#### 🖐 Step 1.3.1: ベースConfigをコピー
```bash
cd /home/user/TreeFormer
cp configs/tree_2D_use_mst_only1.yaml configs/tree_2D_guyot_test.yaml
```

#### 🖐 Step 1.3.2: DATA_PATH変更
```bash
# configs/tree_2D_guyot_test.yaml の DATA.DATA_PATH を変更
# Before: DATA_PATH: './data/toulouse-road-network'
# After:  DATA_PATH: './data/guyot_200_20_resized'
```

**Editコマンド例**:
```python
# Edit tool を使用
old: "  DATA_PATH: './data/toulouse-road-network'"
new: "  DATA_PATH: './data/guyot_200_20_resized'"
```

#### 🖐 Step 1.3.3: DATASET名変更
```bash
# configs/tree_2D_guyot_test.yaml の DATA.DATASET を変更
# Before: DATASET: 'toulouse-road-network-2D'
# After:  DATASET: 'guyot-2D'
```

**Editコマンド例**:
```python
old: "  DATASET: 'toulouse-road-network-2D'"
new: "  DATASET: 'guyot-2D'"
```

#### 🖐 Step 1.3.4: EPOCHS変更（テスト用）
```bash
# configs/tree_2D_guyot_test.yaml の TRAIN.EPOCHS を変更
# Before: EPOCHS: 1000
# After:  EPOCHS: 2
```

**Editコマンド例**:
```python
old: "  EPOCHS: 1000"
new: "  EPOCHS: 2"
```

#### 🖐 Step 1.3.5: exp_name変更
```bash
# configs/tree_2D_guyot_test.yaml の log.exp_name を変更
# Before: exp_name: 'experiment_use_mst_paper_8_data_rotate100'
# After:  exp_name: 'test_guyot_baseline_nov14_2025'
```

**Editコマンド例**:
```python
old: "  exp_name: 'experiment_use_mst_paper_8_data_rotate100'"
new: "  exp_name: 'test_guyot_baseline_nov14_2025'"
```

#### 🔎 Step 1.3.6: 作成したConfigの確認
```bash
cat configs/tree_2D_guyot_test.yaml
```

**確認項目**:
- DATA_PATH: './data/guyot_200_20_resized'
- DATASET: 'guyot-2D'
- EPOCHS: 2
- exp_name: 'test_guyot_baseline_nov14_2025'

#### 🧪 Step 1.3.7: 新Config読み込みテスト
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_config_load.py configs/tree_2D_guyot_test.yaml
```

**成功条件**:
- DATA_PATHに`guyot_200_20_resized`が表示
- DATASETに`guyot-2D`が表示
- EPOCHSに`2`が表示

---

### Task 1.4: Config差分確認テスト

#### 🔎 Step 1.4.1: 差分表示
```bash
diff -u configs/tree_2D_use_mst_only1.yaml configs/tree_2D_guyot_test.yaml
```

**確認項目**: 以下4箇所のみが変更されていること
- DATA_PATH
- DATASET
- EPOCHS
- exp_name

#### 🧪 Step 1.4.2: Validation scriptによる確認
```bash
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
```

#### 🧪 Step 1.4.3: Validation実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_config_diff.py
```

**成功条件**: `✓ All config changes validated successfully`が出力

#### 🛠 Step 1.4.4: エラー時対処 - Assertion失敗
**症状**: `AssertionError: DATA_PATH not updated`等

**対処**:
1. エラーメッセージから該当箇所を特定
2. Step 1.3.2〜1.3.5の該当ステップを再実行
3. 再度Step 1.4.3を実行

---

## Phase 2: Dataset Loader動作確認

### ✅ チェックリスト
- [ ] 2.1 Dataset loaderコード解析
- [ ] 2.2 Guyot dataset class存在確認
- [ ] 2.3 Dataset読み込みテスト実行
- [ ] 2.4 Batch取得テスト

---

### Task 2.1: Dataset loaderコード解析

#### 🖐 Step 2.1.1: train_mst.py内のdataset import確認
```bash
grep -n "import.*dataset" train_mst.py | head -20
```

**期待結果**: dataset関連のimport文が表示される

#### 🖐 Step 2.1.2: dataset初期化部分の特定
```bash
grep -n "dataset.*=" train_mst.py | grep -i "train\|test\|val" | head -20
```

**期待結果**: dataset objectが作成される行番号が表示される

#### 🔎 Step 2.1.3: dataset初期化コードの詳細確認
```bash
# 前ステップで特定した行番号周辺を確認
# 例: 行番号が500の場合
sed -n '490,510p' train_mst.py
```

**確認項目**:
- dataset classの種類（ToulouseDataset? GuyotDataset?）
- 初期化時のパラメータ
- config['DATA']['DATASET']の使用方法

#### 🖐 Step 2.1.4: データセットクラスファイルの検索
```bash
find . -type f -name "*.py" -exec grep -l "class.*Dataset" {} \; | grep -v __pycache__ | head -20
```

**期待結果**: dataset定義ファイルのリスト

#### 🔎 Step 2.1.5: Guyot dataset class検索
```bash
find . -type f -name "*.py" -exec grep -l "guyot\|Guyot" {} \; | grep -v __pycache__
```

**確認項目**: Guyot専用のdataset classが存在するか

#### 🛠 Step 2.1.6: エラー時対処 - Dataset class不明
**症状**: Guyot用dataset classが見つからない

**対処**:
```bash
# Generic dataset classを探す
grep -rn "class.*Dataset.*torch" --include="*.py" | grep -v __pycache__ | head -10
# 見つかったファイルを精読し、柔軟なdataset loaderか確認
```

---

### Task 2.2: Guyot dataset class存在確認

#### 🖐 Step 2.2.1: datasets/ディレクトリ確認
```bash
ls -la datasets/ 2>/dev/null || echo "datasets/ directory not found"
```

#### 🖐 Step 2.2.2: 代替ディレクトリ検索
```bash
find . -type d -name "*data*" | grep -v ".venv\|__pycache__\|.git" | head -10
```

#### 🔎 Step 2.2.3: Dataset実装ファイルの特定
```bash
# train_mst.pyから実際にimportされているmoduleを確認
grep "^from.*import.*" train_mst.py | grep -i "data\|dataset" | head -10
```

#### 🧪 Step 2.2.4: Dataset import テスト
```bash
cat > /tmp/test_dataset_import.py << 'EOF'
import sys
sys.path.insert(0, '/home/user/TreeFormer')

# train_mst.pyから抽出したimport文を順次試す
# 例:
try:
    from datasets.guyot_dataset import GuyotDataset
    print("✓ GuyotDataset found")
except ImportError as e:
    print(f"✗ GuyotDataset not found: {e}")

try:
    from datasets.tree_dataset import TreeDataset
    print("✓ TreeDataset found")
except ImportError as e:
    print(f"✗ TreeDataset not found: {e}")

# 他の可能性も追加
EOF
```

#### 🧪 Step 2.2.5: Import テスト実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_dataset_import.py
```

**成功条件**: 少なくとも1つのDataset classが正常にimportされる

#### 🛠 Step 2.2.6: エラー時対処 - Dataset未実装
**症状**: 全てのDataset importが失敗

**対処手順**:
1. train_mst.pyの実際の実装を精読
2. どのようにdataを読み込んでいるか確認
3. 必要に応じてGeneric dataset classの実装を検討
4. **別途Issue作成を検討**: "Implement Guyot dataset loader"

---

### Task 2.3: Dataset読み込みテスト実行

#### 🖐 Step 2.3.1: Minimal dataset load script作成
```bash
cat > /tmp/test_dataset_load.py << 'EOF'
import sys
sys.path.insert(0, '/home/user/TreeFormer')
import yaml
import torch

# Config読み込み
with open('configs/tree_2D_guyot_test.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Dataset path: {config['DATA']['DATA_PATH']}")
print(f"Dataset name: {config['DATA']['DATASET']}")

# Dataset初期化（実際のコードに合わせて調整）
# 以下はプレースホルダー、実際のtrain_mst.pyの実装に置き換える
# from datasets.xxx import XXXDataset
# dataset = XXXDataset(config)
# print(f"Dataset size: {len(dataset)}")

print("NOTE: Actual dataset loading code needs to be implemented based on train_mst.py")
EOF
```

#### 🔎 Step 2.3.2: train_mst.pyのdataset初期化コードをコピー
```bash
# train_mst.pyから実際のdataset初期化部分を抽出
# 例: 500-520行目にある場合
sed -n '500,520p' train_mst.py > /tmp/dataset_init_snippet.py
cat /tmp/dataset_init_snippet.py
```

#### 🖐 Step 2.3.3: 抽出コードを/tmp/test_dataset_load.pyに統合
**手動作業**:
1. /tmp/dataset_init_snippet.pyの内容を確認
2. 必要なimport文を/tmp/test_dataset_load.pyに追加
3. dataset初期化コードを追加
4. print文でdataset情報を出力

#### 🧪 Step 2.3.4: Dataset読み込みテスト実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_dataset_load.py
```

**成功条件**:
- `Dataset size: 200` (trainの場合)
- エラーなく完了

#### 🛠 Step 2.3.5: エラー時対処 - FileNotFoundError
**症状**: `FileNotFoundError: data/guyot_200_20_resized/...`

**対処**:
```bash
# パスの確認
ls -la data/guyot_200_20_resized/
# Configのパスが正しいか再確認
grep DATA_PATH configs/tree_2D_guyot_test.yaml
# 相対パスの問題の可能性 → 絶対パスで試す
```

#### 🛠 Step 2.3.6: エラー時対処 - ImportError
**症状**: `ImportError: cannot import name 'XXX'`

**対処**:
```bash
# 依存関係の確認
pip list | grep -i torch
pip list | grep -i numpy
# 不足しているpackageをインストール
pip install <missing_package>
```

---

### Task 2.4: Batch取得テスト

#### 🧪 Step 2.4.1: DataLoader作成テストスクリプト
```bash
cat > /tmp/test_dataloader.py << 'EOF'
import sys
sys.path.insert(0, '/home/user/TreeFormer')
import yaml
import torch
from torch.utils.data import DataLoader

# Config読み込み
with open('configs/tree_2D_guyot_test.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Dataset初期化（前ステップで確認したコードを使用）
# from datasets.xxx import XXXDataset
# dataset = XXXDataset(config)

# DataLoader作成
# dataloader = DataLoader(
#     dataset,
#     batch_size=config['DATA']['BATCH_SIZE'],
#     num_workers=config['DATA']['NUM_WORKERS'],
#     shuffle=True
# )

# 1 batch取得テスト
# for batch in dataloader:
#     print(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'tensor'}")
#     if isinstance(batch, dict):
#         for key, value in batch.items():
#             if isinstance(value, torch.Tensor):
#                 print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
#     break

print("NOTE: Implement actual DataLoader code based on train_mst.py")
EOF
```

#### 🖐 Step 2.4.2: train_mst.pyのDataLoader部分を抽出
```bash
grep -n "DataLoader" train_mst.py | head -10
# 該当行周辺を確認
sed -n '<行番号-10>,<行番号+10>p' train_mst.py
```

#### 🖐 Step 2.4.3: DataLoaderコードを統合
**手動作業**: /tmp/test_dataloader.pyに実際のDataLoaderコードを追加

#### 🧪 Step 2.4.4: DataLoader実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/test_dataloader.py
```

**成功条件**:
- Batch keysが表示される
- 各tensorのshapeとdtypeが表示される
- エラーなく完了

#### 🔎 Step 2.4.5: Batch内容の妥当性確認
**確認項目**:
- Image tensor: shape=(batch_size, 3, 512, 512)程度
- Annotation関連: nodes, edgesなどのkey存在
- dtype: torch.float32 (image), torch.long (labels)など適切

#### 🛠 Step 2.4.6: エラー時対処 - Shape不一致
**症状**: `RuntimeError: stack expects each tensor to be equal size`

**対処**:
```bash
# collate_fn が必要な可能性
# train_mst.pyでcollate_fnが使用されているか確認
grep -n "collate" train_mst.py
# カスタムcollate_fnの実装を確認し、DataLoaderに追加
```

---

## Phase 3: 依存関係確認

### ✅ チェックリスト
- [ ] 3.1 requirements/pyproject.toml確認
- [ ] 3.2 現在の環境確認
- [ ] 3.3 不足package特定
- [ ] 3.4 追加packageインストール

---

### Task 3.1: requirements/pyproject.toml確認

#### 🖐 Step 3.1.1: requirements.txt存在確認
```bash
ls -la requirements.txt 2>/dev/null || echo "requirements.txt not found"
```

#### 🖐 Step 3.1.2: pyproject.toml確認
```bash
ls -la pyproject.toml 2>/dev/null || echo "pyproject.toml not found"
```

#### 🔎 Step 3.1.3: 依存関係ファイル内容確認
```bash
if [ -f requirements.txt ]; then
    cat requirements.txt
elif [ -f pyproject.toml ]; then
    cat pyproject.toml
else
    echo "No dependency file found"
fi
```

#### 🖐 Step 3.1.4: train_mst.py内のimport一覧抽出
```bash
grep "^import\|^from.*import" train_mst.py | sort -u
```

**確認項目**: 必要なpackage一覧
- torch, torchvision
- numpy, scipy
- networkx
- opencv (cv2)
- PIL
- yaml
- tqdm
- matplotlib
- など

---

### Task 3.2: 現在の環境確認

#### 🔎 Step 3.2.1: 仮想環境の確認
```bash
source /home/user/TreeFormer/.venv/bin/activate
which python
python --version
```

**成功条件**:
- pythonが`.venv/bin/python`を指す
- Python 3.8以上

#### 🔎 Step 3.2.2: インストール済みpackage一覧
```bash
pip list
```

#### 🔎 Step 3.2.3: 主要package確認
```bash
pip list | grep -E "torch|numpy|scipy|networkx|opencv|Pillow|PyYAML|tqdm|matplotlib"
```

#### 🧪 Step 3.2.4: 主要packageインポートテスト
```bash
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

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError as e:
        print(f"✗ {pkg}: {e}")
EOF

python /tmp/test_imports.py
```

**成功条件**: 全パッケージに`✓`がつく

---

### Task 3.3: 不足package特定

#### 🔎 Step 3.3.1: 不足package抽出
```bash
# 前ステップの結果から✗のあるpackageをリストアップ
python /tmp/test_imports.py 2>&1 | grep "✗"
```

#### 🖐 Step 3.3.2: 不足packageのリスト作成
```bash
# 例
echo "Missing packages:" > /tmp/missing_packages.txt
python /tmp/test_imports.py 2>&1 | grep "✗" | awk '{print $2}' | sed 's/:$//' >> /tmp/missing_packages.txt
cat /tmp/missing_packages.txt
```

#### 🔎 Step 3.3.3: package名の正規化
**手動作業**:
- `cv2` → `opencv-python`
- `PIL` → `Pillow`
- `yaml` → `PyYAML`
- など、pip install時の正しい名前に変換

---

### Task 3.4: 追加packageインストール

#### 🖐 Step 3.4.1: 不足package一括インストール
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# 例: 不足しているpackageをインストール
# pip install opencv-python Pillow PyYAML networkx scipy

# または、requirements.txtがある場合
# pip install -r requirements.txt
```

#### 🔎 Step 3.4.2: インストール確認
```bash
pip list | grep -E "opencv|Pillow|PyYAML|networkx|scipy"
```

#### 🧪 Step 3.4.3: 再度importテスト
```bash
python /tmp/test_imports.py
```

**成功条件**: 全パッケージに`✓`がつく

#### 🛠 Step 3.4.4: エラー時対処 - インストール失敗
**症状**: `ERROR: Could not find a version that satisfies the requirement`

**対処**:
```bash
# Package名が間違っている可能性
pip search <package_name>  # pip search廃止の場合はpypi.orgで確認
# 正しい名前でリトライ

# または、uvを使用
uv pip install <package_name>
```

#### 🛠 Step 3.4.5: エラー時対処 - 依存関係コンフリクト
**症状**: `ERROR: ... has requirement xxx!=yyy, but you have yyy`

**対処**:
```bash
# 依存関係ツリー確認
pip list --format=freeze | grep <conflicting_package>
# バージョン指定してインストール
pip install <package>==<compatible_version>
```

---

## Phase 4: 最小限Training実行

### ✅ チェックリスト
- [ ] 4.1 Training script引数確認
- [ ] 4.2 DRY RUN実行（0 epoch）
- [ ] 4.3 1 Epoch学習実行
- [ ] 4.4 Checkpoint保存確認
- [ ] 4.5 Log出力確認

---

### Task 4.1: Training script引数確認

#### 🔎 Step 4.1.1: train_mst.pyのヘルプ表示
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python train_mst.py --help
```

**確認項目**:
- `--config`: Config fileパス
- `--resume`: Checkpoint再開
- `--device`: CPU/GPU指定
- `--use_mst_train`: MST訓練フラグ
- その他オプション

#### 🖐 Step 4.1.2: 引数パラメータをメモ
**記録**:
```
Required args:
  --config <path>

Optional args:
  --device cuda|cpu
  --resume <checkpoint_path>
  --use_mst_train (flag)
```

---

### Task 4.2: DRY RUN実行（0 epoch）

#### 🖐 Step 4.2.1: Config一時変更（EPOCHS=0）
```bash
cp configs/tree_2D_guyot_test.yaml configs/tree_2D_guyot_dry_run.yaml
# EPOCHS: 2 → 0に変更
sed -i 's/EPOCHS: 2/EPOCHS: 0/' configs/tree_2D_guyot_dry_run.yaml
```

#### 🔎 Step 4.2.2: 変更確認
```bash
grep "EPOCHS:" configs/tree_2D_guyot_dry_run.yaml
```

**成功条件**: `EPOCHS: 0`が表示

#### 🧪 Step 4.2.3: DRY RUN実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# GPU利用可能な場合
python train_mst.py --config configs/tree_2D_guyot_dry_run.yaml --device cuda

# GPUなしの場合
# python train_mst.py --config configs/tree_2D_guyot_dry_run.yaml --device cpu
```

**期待動作**:
- Config読み込み成功
- Dataset読み込み成功
- Model初期化成功
- Epoch 0なので学習はスキップ
- エラーなく終了

#### 🔎 Step 4.2.4: 出力ログ確認
**確認項目**:
- `Loading config from ...`
- `Dataset size: 200`
- `Model initialized`
- No error messages

#### 🛠 Step 4.2.5: エラー時対処 - CUDA out of memory
**症状**: `RuntimeError: CUDA out of memory`

**対処**:
```bash
# Batch size削減
# configs/tree_2D_guyot_dry_run.yaml の BATCH_SIZE: 8 → 4または2に変更
sed -i 's/BATCH_SIZE: 8/BATCH_SIZE: 4/' configs/tree_2D_guyot_dry_run.yaml

# 再実行
python train_mst.py --config configs/tree_2D_guyot_dry_run.yaml --device cuda
```

#### 🛠 Step 4.2.6: エラー時対処 - ModuleNotFoundError
**症状**: `ModuleNotFoundError: No module named 'xxx'`

**対処**:
```bash
# Phase 3に戻り、不足packageをインストール
pip install xxx
# 再度DRY RUN
```

---

### Task 4.3: 1 Epoch学習実行

#### 🖐 Step 4.3.1: 1 Epoch config作成
```bash
cp configs/tree_2D_guyot_test.yaml configs/tree_2D_guyot_1epoch.yaml
# EPOCHS: 2 → 1に変更
sed -i 's/EPOCHS: 2/EPOCHS: 1/' configs/tree_2D_guyot_1epoch.yaml
# exp_name変更
sed -i 's/test_guyot_baseline_nov14_2025/test_guyot_1epoch_nov14_2025/' configs/tree_2D_guyot_1epoch.yaml
```

#### 🔎 Step 4.3.2: Config確認
```bash
grep -E "EPOCHS:|exp_name:" configs/tree_2D_guyot_1epoch.yaml
```

**成功条件**:
- `EPOCHS: 1`
- `exp_name: 'test_guyot_1epoch_nov14_2025'`

#### 🧪 Step 4.3.3: 1 Epoch学習実行
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# 実行時間を計測
time python train_mst.py --config configs/tree_2D_guyot_1epoch.yaml --device cuda
```

**期待動作**:
- Dataset読み込み: 200 samples
- Training開始
- Batch毎のloss表示
- 1 epoch完了
- Checkpoint保存
- 正常終了

#### 🔎 Step 4.3.4: 実行中のログ監視
**別ターミナル**:
```bash
# Logファイルが生成されている場合
tail -f trained_weights/test_guyot_1epoch_nov14_2025/train.log
```

**確認項目**:
- Epoch 1/1
- Iteration毎のloss値
- Loss値が発散していないか（NaN, Inf無し）

---

### Task 4.4: Checkpoint保存確認

#### 🔎 Step 4.4.1: 保存先ディレクトリ確認
```bash
ls -la trained_weights/
```

**期待結果**: `test_guyot_1epoch_nov14_2025/`ディレクトリが存在

#### 🔎 Step 4.4.2: Checkpoint内容確認
```bash
ls -lh trained_weights/test_guyot_1epoch_nov14_2025/
```

**確認項目**:
- `checkpoint_*.pth` または `model_*.pth`
- `train.log`
- その他設定ファイル

#### 🧪 Step 4.4.3: Checkpoint読み込みテスト
```bash
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

**成功条件**:
- `✓ Checkpoint loaded successfully`
- Keys に`epoch`, `model_state_dict`等が含まれる

#### 🛠 Step 4.4.4: エラー時対処 - Checkpoint未保存
**症状**: `✗ No checkpoint found`

**対処**:
```bash
# Configの SAVE_PATH 確認
grep SAVE_PATH configs/tree_2D_guyot_1epoch.yaml
# train_mst.py内のcheckpoint保存コード確認
grep -n "save.*checkpoint\|torch.save" train_mst.py | head -10
# 保存条件（epoch間隔等）を確認し、必要に応じてConfig調整
```

---

### Task 4.5: Log出力確認

#### 🔎 Step 4.5.1: Log fileの内容表示
```bash
cat trained_weights/test_guyot_1epoch_nov14_2025/train.log 2>/dev/null || echo "train.log not found"
```

#### 🔎 Step 4.5.2: Loss値の抽出
```bash
grep -i "loss" trained_weights/test_guyot_1epoch_nov14_2025/train.log | tail -20
```

**確認項目**:
- Total loss, node loss, edge loss等の値
- Loss値が数値として妥当（NaN/Infでない）
- Epoch終了時のloss summary

#### 🧪 Step 4.5.3: Loss推移の可視化準備
```bash
cat > /tmp/plot_loss.py << 'EOF'
import re
import matplotlib.pyplot as plt

log_file = 'trained_weights/test_guyot_1epoch_nov14_2025/train.log'

losses = []
with open(log_file, 'r') as f:
    for line in f:
        # 例: "Iteration 10: loss=1.234"
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
else:
    print("✗ No loss values found in log")
EOF

cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/plot_loss.py
```

**成功条件**:
- `/tmp/loss_plot.png`が生成される
- Initial loss > Final loss（学習が進んでいる）

#### 🔎 Step 4.5.4: Plot画像確認
```bash
file /tmp/loss_plot.png
# ファイルマネージャで開いて目視確認
```

---

## Phase 5: エラートラブルシューティング

### ✅ チェックリスト
- [ ] 5.1 Dataset関連エラー対処
- [ ] 5.2 Model関連エラー対処
- [ ] 5.3 Memory関連エラー対処
- [ ] 5.4 Loss関連エラー対処

---

### Task 5.1: Dataset関連エラー対処

#### 🛠 Step 5.1.1: Dataset class不一致エラー
**症状**: `AttributeError: 'NoneType' object has no attribute 'dataset'`

**対処**:
1. train_mst.py内のdataset初期化コードを確認
2. CONFIG['DATA']['DATASET']の値を確認
3. Dataset classのmappingを確認
4. 必要に応じてtrain_mst.py内のdataset選択ロジックを修正

#### 🛠 Step 5.1.2: Annotation format不一致
**症状**: `KeyError: 'VineFeature'`等

**対処**:
```bash
# Annotation構造の確認
python -c "import json; print(json.dumps(json.load(open('data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001_annotation.json')), indent=2))" | head -50

# Dataset classの期待するformat確認
# → Dataset class内のannotation読み込み部分を修正
```

#### 🛠 Step 5.1.3: 画像読み込みエラー
**症状**: `OSError: cannot identify image file`

**対処**:
```bash
# 問題の画像を特定
find data/guyot_200_20_resized/ -name "*.jpeg" -exec file {} \; | grep -v JPEG

# 破損画像を除外またはスキップする処理をdataset classに追加
```

---

### Task 5.2: Model関連エラー対処

#### 🛠 Step 5.2.1: Model初期化エラー
**症状**: `RuntimeError: Error(s) in loading state_dict`

**対処**:
```bash
# Config内のMODELパラメータ確認
grep -A 50 "^MODEL:" configs/tree_2D_guyot_1epoch.yaml

# Pre-trained weightを使用している場合、arch一致確認
# 不一致の場合、--resumeオプションを外してscratch学習
```

#### 🛠 Step 5.2.2: Input shape不一致
**症状**: `RuntimeError: Expected 4D tensor, got 3D`

**対処**:
```bash
# Dataset出力のshape確認
# /tmp/test_dataloader.pyで確認した内容を再チェック
# Transformの追加が必要な場合、dataset classに追加
```

---

### Task 5.3: Memory関連エラー対処

#### 🛠 Step 5.3.1: CUDA OOM
**症状**: `RuntimeError: CUDA out of memory`

**対処**:
```bash
# Batch size削減
sed -i 's/BATCH_SIZE: 8/BATCH_SIZE: 2/' configs/tree_2D_guyot_1epoch.yaml

# NUM_WORKERS削減
sed -i 's/NUM_WORKERS: 4/NUM_WORKERS: 2/' configs/tree_2D_guyot_1epoch.yaml

# 再実行
```

#### 🛠 Step 5.3.2: CPU Memory不足
**症状**: `MemoryError`

**対処**:
```bash
# NUM_WORKERS=0に設定（multi-process無効化）
sed -i 's/NUM_WORKERS: [0-9]/NUM_WORKERS: 0/' configs/tree_2D_guyot_1epoch.yaml

# Dataset一部のみ使用
# → train_mst.py内でsubset作成コード追加を検討
```

---

### Task 5.4: Loss関連エラー対処

#### 🛠 Step 5.4.1: Loss NaN/Inf
**症状**: Loss値がNaN or Inf

**対処**:
```bash
# Learning rate削減
sed -i 's/LR: 1e-4/LR: 1e-5/' configs/tree_2D_guyot_1epoch.yaml

# Gradient clipping確認
grep CLIP_MAX_NORM configs/tree_2D_guyot_1epoch.yaml
# → 値を小さく（例: 0.1 → 0.05）

# 再実行
```

#### 🛠 Step 5.4.2: Loss発散
**症状**: Loss値が増加し続ける

**対処**:
```bash
# データ正規化確認
# → Dataset class内のtransformにNormalize追加

# Loss weight調整
# W_BBOX, W_CLASS等の値を小さく
sed -i 's/W_BBOX: 2.0/W_BBOX: 1.0/' configs/tree_2D_guyot_1epoch.yaml
```

---

## Phase 6: 成功基準検証

### ✅ チェックリスト
- [ ] 6.1 Training完走確認
- [ ] 6.2 Checkpoint生成確認
- [ ] 6.3 Loss収束傾向確認
- [ ] 6.4 最終レポート作成

---

### Task 6.1: Training完走確認

#### 🔎 Step 6.1.1: 1 Epoch完了の確認
```bash
grep -i "epoch.*1.*complete\|finished" trained_weights/test_guyot_1epoch_nov14_2025/train.log
```

**成功条件**: Epoch 1完了のメッセージが存在

#### 🔎 Step 6.1.2: 全iteration実行確認
```bash
# 予想iteration数 = dataset_size / batch_size
# 200 / 8 = 25 iterations (batch_size=8の場合)
grep -c "Iteration\|Step" trained_weights/test_guyot_1epoch_nov14_2025/train.log
```

**成功条件**: 25前後のiteration数

---

### Task 6.2: Checkpoint生成確認

#### 🔎 Step 6.2.1: Checkpoint存在確認
```bash
find trained_weights/test_guyot_1epoch_nov14_2025/ -name "*.pth" -ls
```

**成功条件**: 少なくとも1つの.pthファイルが存在

#### 🧪 Step 6.2.2: Checkpoint読み込み・再開テスト
```bash
# 2 epoch目を追加実行してresume機能確認
cp configs/tree_2D_guyot_1epoch.yaml configs/tree_2D_guyot_resume_test.yaml
sed -i 's/EPOCHS: 1/EPOCHS: 2/' configs/tree_2D_guyot_resume_test.yaml

# Resume実行
cd /home/user/TreeFormer
source .venv/bin/activate

CHECKPOINT_PATH=$(find trained_weights/test_guyot_1epoch_nov14_2025/ -name "*.pth" | head -1)
python train_mst.py --config configs/tree_2D_guyot_resume_test.yaml --resume $CHECKPOINT_PATH --device cuda
```

**成功条件**:
- Epoch 2から開始
- エラーなく完了

---

### Task 6.3: Loss収束傾向確認

#### 🔎 Step 6.3.1: Initial vs Final loss比較
```bash
cd /home/user/TreeFormer
source .venv/bin/activate
python /tmp/plot_loss.py
```

**成功条件**:
- Final loss < Initial loss
- Loss値が数値として妥当（0.01〜100程度、NaN/Infでない）

#### 🔎 Step 6.3.2: Loss plot目視確認
```bash
# /tmp/loss_plot.pngを開く
xdg-open /tmp/loss_plot.png 2>/dev/null || echo "Manually open /tmp/loss_plot.png"
```

**確認項目**:
- 右下がりの傾向
- 大きな発散無し

---

### Task 6.4: 最終レポート作成

#### 🖐 Step 6.4.1: 検証結果サマリ作成
```bash
cat > /tmp/training_verification_report.md << 'EOF'
# TreeFormer Guyot Dataset Training Verification Report
**Date**: 2025-11-14
**Config**: configs/tree_2D_guyot_1epoch.yaml
**Branch**: claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS

## Summary
- ✓/✗ Dataset loading:
- ✓/✗ 1 Epoch training completion:
- ✓/✗ Checkpoint generation:
- ✓/✗ Loss convergence:

## Dataset
- Train samples: 200
- Test samples: 20
- Image size: 512x512
- Format: JPEG + JSON annotations

## Training Results
- Epochs completed: 1
- Total iterations: <記入>
- Initial loss: <記入>
- Final loss: <記入>
- Training time: <記入>

## Files Generated
- Checkpoint: trained_weights/test_guyot_1epoch_nov14_2025/*.pth
- Log: trained_weights/test_guyot_1epoch_nov14_2025/train.log
- Loss plot: /tmp/loss_plot.png

## Issues Encountered
<発生した問題と対処法を記載>

## Next Steps
1. Full training (100+ epochs) with validation
2. Hyperparameter tuning
3. Evaluation on test set
4. Comparison with baseline

## Conclusion
Guyot datasetでのTreeFormer学習が正常に動作することを確認した。
EOF
```

#### 🖐 Step 6.4.2: 実測値の記入
**手動作業**:
1. /tmp/training_verification_report.mdを開く
2. <記入>箇所に実際の値を入力
3. Issues Encounteredに問題と対処を記載

#### 🔎 Step 6.4.3: レポート内容確認
```bash
cat /tmp/training_verification_report.md
```

#### 🖐 Step 6.4.4: レポートをリポジトリに保存
```bash
cp /tmp/training_verification_report.md /home/user/TreeFormer/temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md
```

#### 🔎 Step 6.4.5: Git commit準備
```bash
cd /home/user/TreeFormer
git status
git add temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md
git add configs/tree_2D_guyot_*.yaml
```

**確認項目**: 追加されたファイル一覧

---

## 📊 全体進捗管理

### Phase完了チェック
- [ ] Phase 1: Dataset & Config 事前調査
- [ ] Phase 2: Dataset Loader動作確認
- [ ] Phase 3: 依存関係確認
- [ ] Phase 4: 最小限Training実行
- [ ] Phase 5: エラートラブルシューティング (必要に応じて)
- [ ] Phase 6: 成功基準検証

### 最終成果物
1. **Config files**:
   - configs/tree_2D_guyot_test.yaml
   - configs/tree_2D_guyot_1epoch.yaml

2. **Test scripts**:
   - /tmp/test_image_load.py
   - /tmp/test_config_load.py
   - /tmp/test_dataset_load.py
   - /tmp/test_dataloader.py
   - /tmp/test_checkpoint.py
   - /tmp/plot_loss.py

3. **Training artifacts**:
   - trained_weights/test_guyot_1epoch_nov14_2025/*.pth
   - trained_weights/test_guyot_1epoch_nov14_2025/train.log

4. **Documentation**:
   - temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md
   - temp/WORK_PLAN_nov14_2025_training_verification.md (本ファイル)

---

## 🔄 Parallel Execution可能箇所

以下のTask Groupは並行実行可能:

### Group A (Phase 1初期調査)
- Task 1.1: Dataset実ファイル確認
- Task 1.2: 既存Config全パラメータ解析

### Group B (Phase 2コード解析)
- Task 2.1: Dataset loaderコード解析
- Task 3.1: requirements/pyproject.toml確認

### Group C (検証スクリプト作成)
- Task 1.1.7: 画像読み込みテストスクリプト作成
- Task 1.2.5: Config読み込みテストスクリプト作成
- Task 2.3.1: Minimal dataset load script作成

**注意**: Phaseをまたぐ並行実行は依存関係に注意

---

## 📝 作業記録欄

### 作業開始時刻
- [ ] 記入: YYYY-MM-DD HH:MM

### Phase別完了時刻
- [ ] Phase 1完了:
- [ ] Phase 2完了:
- [ ] Phase 3完了:
- [ ] Phase 4完了:
- [ ] Phase 5実施: (必要に応じて)
- [ ] Phase 6完了:

### 問題発生記録
| 時刻 | Phase/Task | 問題内容 | 対処 | 結果 |
|------|-----------|---------|------|------|
|      |           |         |      |      |

### 備考
<気づき・改善点・次回への申し送り事項>

---

**END OF WORK PLAN**
