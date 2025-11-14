# 環境セットアップ・オンボーディングガイド

**作成日**: 2025-11-14
**対象**: 新しいセッション・エージェント、開発メンバー
**プロジェクト**: TreeFormer (yuki-inaho/TreeFormer)
**目的**: TreeFormerの開発・学習環境を迅速に再現し、既存作業を継続できるようにする

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [現在のプロジェクト状態](#2-現在のプロジェクト状態)
3. [前提条件の確認](#3-前提条件の確認)
4. [環境セットアップ手順](#4-環境セットアップ手順)
5. [動作確認](#5-動作確認)
6. [トラブルシューティング](#6-トラブルシューティング)
7. [次のステップ](#7-次のステップ)
8. [環境セットアップ完了チェックリスト](#8-環境セットアップ完了チェックリスト)
9. [更新履歴](#9-更新履歴)

---

## 1. プロジェクト概要

### プロジェクト名

**TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation**

深層学習を用いた単一画像からの植物骨格（tree structure）推定フレームワーク。Deformable Transformerベースのモデルで、グラフ生成とMST（Minimum Spanning Tree）制約を組み合わせて木構造を抽出します。

### 最終目標

Guyot grapevine dataset（ブドウの木の画像データセット）を用いて、TreeFormerモデルを学習・評価し、高精度な植物骨格推定を実現する。3つのアプローチ（TreeFormer w/ SFS Layer、Test-time constraint、Unconstrained baseline）を実装・比較し、論文再現性を確保する。

### 主要コンポーネント

* **Training Pipeline**: train_mst.py / train_unmst.py - MST制約あり/なしの学習
* **Inference Scripts**: inference_infinity_*.py - 3つのアプローチの推論実装
* **Dataset Loaders**: Guyot grapevine dataset対応のデータローダー
* **Validation Scripts**: valid_smd_guyot_nx.py - 検証・評価スクリプト
* **Configuration**: YAML-based configs (configs/tree_2D_*.yaml)
* **Documentation**: temp/以下の作業計画書、調査レポート、オンボーディング資料

---

## 2. 現在のプロジェクト状態

### 完了済み

| 分類                      | 状態 | 説明                                                                          |
| ----------------------- | -- | --------------------------------------------------------------------------- |
| **欠落inference files実装** | ✅  | inference_infinity_mst_nx_gradmst.py等3ファイルを実装済み                             |
| **テストインフラ構築**          | ✅  | tests/conftest.py作成、pytest fixtures実装済み                                     |
| **作業計画書**              | ✅  | temp/WORK_PLAN_nov14_2025_missing_inference_files.md 作成済み                    |
| **Training検証計画書**       | ✅  | temp/WORK_PLAN_nov14_2025_training_verification.md（6 Phase、105ステップ）作成済み     |
| **.gitignore整備**        | ✅  | Python artifacts、backup files等を除外設定済み                                       |
| **アルゴリズム調査**           | ✅  | temp/algorithm_mapping_and_testing_guide.md（論文3アプローチの詳細解析）作成済み             |
| **Dataset README**      | ✅  | data/README.md（Guyot dataset構造説明）整備済み                                      |
| **Config examples**     | ✅  | configs/tree_2D_use_mst_only1.yaml（Toulouse用）存在、Guyot用はこれから作成予定             |
| **Git branch**          | ✅  | claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS（全変更commit・push済み）           |
| **オンボーディングガイド**        | ✅  | temp/ONBOARDING_nov14_2025.md（本ドキュメント）                                      |

### 依存パッケージのインストール状態

**重要**: 新しいセッションでは依存パッケージが未インストールの可能性があります。セクション 4.2で必ず`uv sync`または`pip install`を実行してください。

主要な依存関係:
- **深層学習**: torch, torchvision, mmcv
- **グラフ処理**: networkx, scipy
- **画像処理**: opencv-python, Pillow, scikit-image
- **その他**: numpy, pyyaml, tqdm, matplotlib

### 未実装・これから着手する項目

* Guyot dataset用Config作成（configs/tree_2D_guyot_test.yaml）
* Guyot dataset loaderの動作確認・必要に応じた修正
* 1 Epoch学習テスト実行
* Training検証レポート作成
* 本格的な学習実行（100+ epochs）
* Test setでの評価
* 3アプローチの性能比較

### 重要なファイル／ディレクトリ

```text
/home/user/TreeFormer/
├── README.md                                      # プロジェクト概要
├── .gitignore                                     # Git除外設定
├── train_mst.py                                   # MST制約あり学習スクリプト
├── train_unmst.py                                 # MST制約なし学習スクリプト
├── valid_smd_guyot_nx.py                          # 検証スクリプト
├── inference_infinity_mst_nx_gradmst.py           # TreeFormer w/ SFS Layer
├── inference_infinity_mst_nx_dist.py              # Test-time constraint
├── inference_infinity_gradmst.py                  # Unconstrained baseline
├── configs/
│   ├── tree_2D_use_mst_only1.yaml                 # Toulouse dataset用Config（参考）
│   └── (tree_2D_guyot_test.yaml)                  # Guyot用Config（作成予定）
├── data/
│   ├── README.md                                  # Dataset構造説明
│   └── guyot_200_20_resized/                      # Guyot dataset（512x512リサイズ版）
│       ├── 01-TrainAndValidationSet/              # 200画像 + annotations
│       ├── 02-IndependentTestSet/                 # 20画像 + annotations
│       └── sampling_metadata.json
├── temp/
│   ├── ONBOARDING_nov14_2025.md                   # 本ドキュメント
│   ├── WORK_PLAN_nov14_2025_training_verification.md  # Training検証作業計画書
│   ├── WORK_PLAN_nov14_2025_missing_inference_files.md  # Inference実装作業計画書
│   ├── algorithm_mapping_and_testing_guide.md     # アルゴリズム調査・テストガイド
│   └── TreeFormer_Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation.md  # 論文MD
├── tests/
│   └── conftest.py                                # Pytest fixtures
├── trained_weights/                               # 学習済みモデル保存先
└── .venv/                                         # Python仮想環境（uv管理）
```

---

## 3. 前提条件の確認

新しいセッションで環境を再現する前に、以下を確認してください。

### 3.1 システム情報の確認

```bash
# OS バージョン確認
cat /etc/os-release | grep -E "^(NAME|VERSION)="

# カーネル確認
uname -r

# CPU コア数・メモリ確認
nproc
free -h

# GPU確認（CUDA利用可能な場合）
nvidia-smi 2>/dev/null || echo "CUDA not available (CPU mode will be used)"

# 作業ディレクトリ確認
pwd
```

**期待される出力例:**

```text
NAME="Ubuntu" (or other Linux distribution)
VERSION="20.04" or later
Linux 4.4.0 or later
4+ cores recommended
8GB+ RAM recommended
/home/user/TreeFormer
```

### 3.2 必須ツールの存在確認

```bash
# Python仮想環境マネージャ
which uv && uv --version

# Python本体
python --version

# Git
git --version

# 画像処理に必要なシステムライブラリ（通常プリインストール済み）
ldconfig -p | grep -E "libjpeg|libpng" | head -2
```

uvが見つからない場合:
```bash
# uvインストール（公式手順）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 3.3 Git ブランチ・コミット確認

```bash
# 現在のブランチ確認
git branch --show-current

# 最新コミット確認
git log --oneline -3

# リモート同期状態確認
git status
```

**期待される出力例:**

```text
claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS
a9be8c2 chore: Remove old work plan file (renamed with date)
905ffcb docs: Add training verification work plan for Guyot dataset
9cc4434 chore: Add .gitignore for Python and project files
```

**別のブランチで作業する場合:**
```bash
# ブランチ切り替え
git fetch origin
git checkout claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS
git pull origin claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS
```

---

## 4. 環境セットアップ手順

### 4.1 システムパッケージの確認／インストール

TreeFormerに必要な主要システムパッケージ:

```bash
# 画像処理ライブラリ（通常プリインストール済み）
apt-get update -qq
apt-get install -y -qq libjpeg-dev libpng-dev || echo "Already installed or not needed"

# OpenCV依存（必要に応じて）
apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 || echo "Already installed"
```

**注意事項:**
- CUDAを使用する場合、nvidia-driverとCUDA toolkitが別途必要（環境により異なる）
- CPU環境でも動作可能（学習速度は低下）

### 4.2 Python 依存パッケージのインストール（最重要）

```bash
cd /home/user/TreeFormer

# 仮想環境の確認
ls -la .venv/ 2>/dev/null || echo ".venv not found, will be created"

# 現在のパッケージ状態確認
source .venv/bin/activate 2>/dev/null || echo "Virtual env not activated yet"
pip list 2>/dev/null | head -10

# uv環境の場合（推奨）
uv sync --no-install-project

# または、requirements.txtがある場合
# pip install -r requirements.txt
```

**主要インストール対象パッケージ:**

* **dependencies:**
  * torch - 深層学習フレームワーク（CUDA版またはCPU版）
  * torchvision - 画像処理用PyTorchライブラリ
  * networkx - グラフ処理（MST構築）
  * scipy - 科学計算（sparse matrix処理等）
  * numpy - 数値計算基盤
  * opencv-python - 画像読み込み・前処理
  * Pillow - 画像I/O
  * scikit-image - 画像処理ユーティリティ
  * PyYAML - Config読み込み
  * tqdm - プログレスバー表示
  * matplotlib - 可視化

* **dev-dependencies:**
  * pytest - テストフレームワーク
  * pytest-cov - テストカバレッジ

**手動インストールが必要な場合:**

```bash
source .venv/bin/activate

pip install torch torchvision networkx scipy numpy opencv-python \
    Pillow scikit-image PyYAML tqdm matplotlib pytest pytest-cov
```

**インストール確認:**

```bash
source .venv/bin/activate
pip list | grep -E "torch|networkx|opencv|numpy|scipy|PIL"
```

### 4.3 CUDA環境の確認（GPU使用時のみ）

```bash
# CUDA利用可能か確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

**期待される出力（CUDA環境）:**
```text
CUDA available: True
CUDA version: 11.8 (or your CUDA version)
```

**CPU環境の場合:**
```text
CUDA available: False
CUDA version: N/A
```
→ 問題なし、CPU modeで動作します

### 4.4 環境変数の設定（任意）

```bash
# CUDA関連（必要に応じて）
export CUDA_VISIBLE_DEVICES=0  # 使用するGPU番号

# OpenCV headless mode（GUI不要の場合）
export QT_QPA_PLATFORM=offscreen

# 再現性確保（乱数シード固定）
export PYTHONHASHSEED=3407
```

**永続化する場合:**
```bash
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export PYTHONHASHSEED=3407' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. 動作確認

環境構築後、最低限以下の確認を行ってください。

### 5.1 Python 依存パッケージ動作確認

```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# 主要ライブラリimportテスト
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'torchvision {torchvision.__version__}')"
python -c "import networkx; print(f'NetworkX {networkx.__version__}')"
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import scipy; print(f'SciPy {scipy.__version__}')"
python -c "import yaml; print('PyYAML OK')"
```

**成功条件**: すべてエラーなく、バージョン番号が表示される

### 5.2 Dataset存在確認

```bash
# Train/Validation set
ls -la data/guyot_200_20_resized/01-TrainAndValidationSet/ | head -5
find data/guyot_200_20_resized/01-TrainAndValidationSet/ -name "*.jpeg" | wc -l

# Test set
ls -la data/guyot_200_20_resized/02-IndependentTestSet/ | head -5
find data/guyot_200_20_resized/02-IndependentTestSet/ -name "*.jpeg" | wc -l
```

**成功条件**:
- Train set: 200画像
- Test set: 20画像

### 5.3 Config読み込みテスト

```bash
python -c "
import yaml
with open('configs/tree_2D_use_mst_only1.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('Config keys:', list(config.keys()))
print('Dataset path:', config['DATA']['DATA_PATH'])
print('Batch size:', config['DATA']['BATCH_SIZE'])
print('Epochs:', config['TRAIN']['EPOCHS'])
"
```

**成功条件**: Config内容が正常に表示される

### 5.4 画像読み込みテスト

```bash
python << 'EOF'
from PIL import Image
import json

# サンプル画像読み込み
img_path = 'data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001.jpeg'
anno_path = 'data/guyot_200_20_resized/01-TrainAndValidationSet/Set04_IMG_0001_annotation.json'

img = Image.open(img_path)
print(f"Image size: {img.size}, mode: {img.mode}")

with open(anno_path, 'r') as f:
    anno = json.load(f)
print(f"Annotation keys: {list(anno.keys())}")
if 'VineFeature' in anno:
    print(f"VineFeature count: {len(anno['VineFeature'])}")
print("✓ Image and annotation loading OK")
EOF
```

**成功条件**:
- Image size: (512, 512)
- mode: RGB
- Annotation keys に VineFeature等が含まれる

### 5.5 Training script引数確認

```bash
python train_mst.py --help | head -20
```

**成功条件**: helpメッセージが表示され、--config, --device等のオプションが確認できる

---

## 6. トラブルシューティング

代表的な問題と対処方法を列挙します。

### 問題1: `uv: command not found`

**原因**: uvがインストールされていない、またはPATHが通っていない
**解決策**:

```bash
# uvインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH設定
source $HOME/.cargo/env

# 確認
uv --version
```

### 問題2: `uv sync` で依存パッケージエラー

**症状**: `Failed to download distributions` 等
**解決策**:

```bash
# ネットワーク確認
ping -c 3 pypi.org

# pip経由で手動インストール
source .venv/bin/activate
pip install torch torchvision networkx scipy numpy opencv-python Pillow scikit-image PyYAML tqdm matplotlib

# 再度uv syncを試す
uv sync --no-install-project
```

### 問題3: PyTorch CUDA版とCPU版の不一致

**症状**: `torch.cuda.is_available()` が False だが、GPU環境のはず
**解決策**:

```bash
# 現在のPyTorchバージョン確認
python -c "import torch; print(torch.__version__)"

# CUDA版が必要な場合、再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# （cu118部分は使用するCUDAバージョンに合わせる）

# 確認
python -c "import torch; print(torch.cuda.is_available())"
```

### 問題4: `ImportError: libGL.so.1: cannot open shared object file`

**症状**: OpenCV実行時のライブラリエラー
**解決策**:

```bash
# システムライブラリインストール
apt-get update && apt-get install -y libgl1-mesa-glx

# または環境変数で回避
export OPENCV_IO_ENABLE_OPENEXR=0
export QT_QPA_PLATFORM=offscreen
```

### 問題5: Dataset not found エラー

**症状**: `FileNotFoundError: data/guyot_200_20_resized/...`
**解決策**:

```bash
# 現在のディレクトリ確認
pwd
# /home/user/TreeFormer であることを確認

# データセット存在確認
ls -la data/
find data/ -name "guyot*" -type d

# 相対パスの問題の場合
cd /home/user/TreeFormer
# スクリプト実行時は必ずプロジェクトルートから
```

### 問題6: Config読み込みエラー

**症状**: `yaml.scanner.ScannerError` 等
**解決策**:

```bash
# YAMLファイルの構文チェック
python -c "import yaml; yaml.safe_load(open('configs/tree_2D_use_mst_only1.yaml'))"

# エラー箇所を特定し、手動修正
# インデント、コロン、引用符に注意
```

### 問題7: Git branch不一致

**症状**: 想定と異なるbranchで作業している
**解決策**:

```bash
# 現在のbranch確認
git branch --show-current

# 正しいbranchへ切り替え
git checkout claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS
git pull origin claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS

# 未コミット変更がある場合、stash
git stash
git checkout <target-branch>
git stash pop
```

---

## 7. 次のステップ

環境セットアップ完了後に実施すべき作業を記載します。

### 7.1 作業計画書・設計書の確認

```bash
# Training検証作業計画書（優先度: 高）
cat temp/WORK_PLAN_nov14_2025_training_verification.md | less

# Inference実装作業計画書（参考）
cat temp/WORK_PLAN_nov14_2025_missing_inference_files.md | less

# アルゴリズム調査・テストガイド（詳細理解用）
cat temp/algorithm_mapping_and_testing_guide.md | less

# Dataset説明
cat data/README.md | less
```

**重要ドキュメント内容:**
- **WORK_PLAN_nov14_2025_training_verification.md**:
  - 6 Phase、105ステップの詳細手順
  - Guyot datasetでの学習動作確認
  - 🖐🔎🧪🛠のマーカーで操作・確認・テスト・エラー対処を明示

- **algorithm_mapping_and_testing_guide.md**:
  - 論文の3アプローチ詳細解析
  - MST損失実装の説明
  - MockモデルとDatasetを使ったテストコード

### 7.2 最初の実務タスク（優先順位順）

#### Priority 0: 環境確認完了
```bash
# セクション5の動作確認をすべて実行
# すべて成功することを確認
```

#### Priority 1: Guyot用Config作成
```bash
# 作業計画書 Phase 1, Task 1.3 を参照
cp configs/tree_2D_use_mst_only1.yaml configs/tree_2D_guyot_test.yaml

# 以下の項目を編集:
# - DATA_PATH: './data/guyot_200_20_resized'
# - DATASET: 'guyot-2D'
# - EPOCHS: 2（テスト用）
# - exp_name: 'test_guyot_baseline_nov14_2025'
```

#### Priority 2: Dataset Loader検証
```bash
# 作業計画書 Phase 2 を参照
# train_mst.py内のdataset loading部分を確認
# テストスクリプトで動作確認
```

#### Priority 3: 依存関係最終確認
```bash
# 作業計画書 Phase 3 を参照
# 不足packageがあればインストール
```

#### Priority 4: 1 Epoch学習テスト
```bash
# 作業計画書 Phase 4 を参照
# 最小限の学習を実行してパイプライン全体の動作確認
```

### 7.3 優先順位付きタスクリスト

1. **[P0] 環境セットアップ完了確認** - セクション8のチェックリスト全項目✓
2. **[P0] 作業計画書精読** - temp/WORK_PLAN_nov14_2025_training_verification.md
3. **[P1] Guyot用Config作成** - Phase 1完了
4. **[P1] Dataset動作確認** - Phase 2完了
5. **[P1] 依存関係確認** - Phase 3完了
6. **[P2] 1 Epoch学習テスト** - Phase 4完了
7. **[P2] 検証レポート作成** - Phase 6完了
8. **[P3] 本格学習実行** - 100+ epochs
9. **[P3] Test set評価** - 精度指標計算
10. **[P4] 3アプローチ比較** - 性能・速度比較レポート

### 7.4 参考ドキュメント一覧

**作業指示系:**
- `temp/WORK_PLAN_nov14_2025_training_verification.md` - Training検証作業計画書（最優先）
- `temp/WORK_PLAN_nov14_2025_missing_inference_files.md` - Inference実装作業計画書
- `temp/ONBOARDING_nov14_2025.md` - 本ドキュメント

**技術解析系:**
- `temp/algorithm_mapping_and_testing_guide.md` - アルゴリズム詳細・テストガイド
- `temp/TreeFormer_Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation.md` - 元論文MD

**データセット系:**
- `data/README.md` - Guyot dataset構造・使用方法

**コード系:**
- `train_mst.py` - メインの学習スクリプト
- `inference_infinity_*.py` - 3アプローチの推論実装
- `valid_smd_guyot_nx.py` - 検証スクリプト

### 7.5 便利なコマンド集（プロジェクト共通）

```bash
# 環境変数確認
env | grep -E "CUDA|PYTHON|PATH" | sort

# 現在のPythonパッケージ一覧
pip list | sort

# Dataset統計
echo "Train images: $(find data/guyot_200_20_resized/01-TrainAndValidationSet/ -name '*.jpeg' | wc -l)"
echo "Test images: $(find data/guyot_200_20_resized/02-IndependentTestSet/ -name '*.jpeg' | wc -l)"

# Config一覧
ls -lh configs/*.yaml

# 学習済みモデル確認
find trained_weights/ -name "*.pth" -exec ls -lh {} \;

# Git状態確認
git status --short
git log --oneline -5

# GPU使用状況（CUDA環境のみ）
watch -n 1 nvidia-smi

# ディスク使用量
du -sh data/ trained_weights/ .venv/
```

---

## 8. 環境セットアップ完了チェックリスト

以下をすべて確認してから、実装・検証作業に進んでください。

- [ ] システム情報確認（Linux環境、4+ cores、8GB+ RAM）
- [ ] 必須ツール存在確認（uv or pip、python 3.8+、git）
- [ ] Git branch確認（claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS）
- [ ] 作業ディレクトリ確認（/home/user/TreeFormer）
- [ ] 依存パッケージインストール完了（torch, networkx, opencv等）
- [ ] 主要ライブラリimport成功（セクション5.1）
- [ ] Dataset存在確認（Train: 200, Test: 20画像）
- [ ] Config読み込み成功（セクション5.3）
- [ ] 画像・アノテーション読み込み成功（セクション5.4）
- [ ] Training script helpメッセージ表示成功（セクション5.5）
- [ ] CUDA環境確認（GPU使用時のみ、CPU modeも可）
- [ ] 作業計画書確認（temp/WORK_PLAN_nov14_2025_training_verification.md精読）
- [ ] オンボーディングガイド理解（本ドキュメント）

**すべて✓が付いたら、セクション7.2の実務タスクに進んでください。**

---

## 9. 更新履歴

* `2025-11-14 初版作成（オンボーディング資料整備、環境セットアップ手順体系化）`

---

## このドキュメントについて

本ガイドは、新しいセッションや新規参加メンバーが、短時間でTreeFormer開発・学習環境を再現し、スムーズに既存タスクを引き継げるようにすることを目的としています。

環境セットアップで問題が発生した場合は、本ドキュメントのトラブルシューティング（セクション6）に加え、以下を参照してください:

- **作業計画書のエラー対処セクション**: temp/WORK_PLAN_nov14_2025_training_verification.md Phase 5
- **GitHubリポジトリ**: https://github.com/yuki-inaho/TreeFormer
- **関連Issue**: GitHub Issue #2 (missing inference files関連)

---

**重要な注意事項:**

1. **Branch管理**: 必ず`claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS`で作業してください
2. **Commit前確認**: .gitignoreで除外されないか確認（__pycache__, *.pth等は除外済み）
3. **Push時**: 必ず`git push -u origin <branch-name>`形式で実行
4. **データセット**: data/guyot_200_20_resized/はgitignore対象外（容量に注意）
5. **学習済みモデル**: trained_weights/以下は.gitignoreで除外（Git管理しない）

**質問・問題があった場合:**

1. まず本ドキュメントのセクション6（トラブルシューティング）を確認
2. 作業計画書のエラー対処手順を確認
3. 解決しない場合は、問題の詳細（エラーメッセージ、実行コマンド、環境情報）をまとめて報告

---

**セットアップ成功を祈ります！🌱**
