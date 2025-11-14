# TreeFormer GPU環境引き継ぎドキュメント
## Guyot Dataset Training Verification - GPU Agent Handover

**作成日時**: 2025-11-14 10:56:01 UTC+0000
**ブランチ**: `claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS`
**前作業者**: CPU環境エージェント
**引き継ぎ先**: GPU環境エージェント

---

## 📋 プロジェクト概要

### 目的
TreeFormerモデルをGuyot Grapevine Dataset（ブドウの木の骨格データセット）で学習させ、正常に動作することを検証する。

### データセット
- **名称**: Guyot 200_20 Resized Dataset
- **場所**: `data/guyot_200_20_resized/`
- **構成**:
  - Train: 200画像 + 200アノテーション（`01-TrainAndValidationSet/`）
  - Test: 20画像 + 20アノテーション（`02-IndependentTestSet/`）
- **形式**:
  - 画像: `.jpeg`（1008x756 → 512x512にリサイズ）
  - アノテーション: `.json`（VineFeature構造）

### モデル
- **TreeFormer**: Deformable Transformer ベースの2D関係推論モデル
- **特徴**: ノード・エッジ検出、グラフ構造生成
- **要件**: **GPU必須**（MultiScaleDeformableAttention CUDA extension使用）

---

## 🎯 作業の目標

### 最終ゴール
1. Guyot datasetで1 Epoch以上の学習を完走
2. Checkpointを正常に保存
3. Lossが収束傾向にあることを確認
4. 検証レポートを作成

### 完了の定義
- [ ] 1 Epoch Training が正常完了（エラーなし）
- [ ] Checkpoint が正常生成・読み込み可能
- [ ] Loss が収束傾向（Final < Initial）
- [ ] 最終レポートが作成され、`temp/`に保存済み

---

## ✅ 完了済み作業（Stage 0-3）

### Stage 0: 事前調査フェーズ（完全並列実行済み）
- ✅ **Task A1**: Dataset実ファイル確認
  - Train: 200画像、Test: 20画像確認済み
  - 画像サイズ: 1008x756（GuyotDatasetで512x512にリサイズ対応済み）
- ✅ **Task A2**: 既存Config全パラメータ解析
  - ベースConfig: `configs/tree_2D_use_mst_only1.yaml`
  - 変更必要箇所特定済み
- ✅ **Task A3**: train_mst.pyコード解析
  - LoadCNNDatasetがGuyotと非互換（.ptファイル期待）
  - **対処済み**: GuyotDataset実装、train_mst.pyに統合
- ✅ **Task A4**: 依存関係確認
  - 不足package特定・インストール完了
  - 現在11パッケージ利用可能

### Stage 1: テストスクリプト作成（完全並列実行済み）
- ✅ **Task B1-B4**: 4つのテストスクリプト作成完了
  - 画像読み込みテスト
  - Config読み込みテスト
  - Dataset importテスト
  - 依存性テスト

### Stage 2: Config・依存関係セットアップ（一部並列実行済み）
- ✅ **Task C1**: Guyot用Config作成
  - `configs/tree_2D_guyot_test.yaml`（EPOCHS=2）
  - `configs/tree_2D_guyot_dry_run.yaml`（EPOCHS=0）
- ✅ **Task C2**: Config差分検証（Python validation通過）
- ✅ **Task C3**: 依存package追加インストール（tqdm, scikit-image）

### Stage 3: Dataset Loader検証（順次実行済み）
- ✅ **Task D1**: Dataset読み込みテスト
  - `guyot_dataset.py`実装完了
  - GuyotDataset正常動作確認（200サンプル読み込み成功）
- ✅ **Task D2**: DataLoader batch取得テスト
  - カスタムcollate_fn実装（可変長nodes/edges対応）
  - Batch取得成功（8画像/batch）

---

## 🛑 中断した作業（Stage 4: GPU環境必須）

### 中断理由
CPU環境では**MultiScaleDeformableAttention CUDA extension**がビルドできないため、Stage 4以降の実行が不可能。

### 試行した対処
1. ✅ torchrun使用（distributed training対応）
2. ✅ gloo backend対応追加（CPU環境用）
3. ❌ MultiScaleDeformableAttention CUDA extensionビルド試行
   - `models/ops/setup.py`実行 → **GPU環境必須エラー**

### 中断時点の状態
- **Dataset**: 完全に準備完了
- **Config**: Guyot用に調整済み
- **train_mst.py**: GuyotDataset対応済み、CPU/GPU自動切り替え対応
- **依存関係**: Python packageは全て導入済み
- **CUDA extension**: 未ビルド（GPU環境で実行必要）

---

## 🚀 GPU環境で実行すべき作業（Stage 4-5）

### 前提条件チェック
```bash
# GPU利用可能確認
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# 期待結果: CUDA available: True, GPU count: 1以上
```

### Step 1: CUDA Extension ビルド（最優先）
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# MultiScaleDeformableAttention CUDA extensionビルド
cd models/ops
python setup.py build install

# ビルド成功確認
python -c "import MultiScaleDeformableAttention as MSDA; print('MSDA imported successfully')"
```

**期待時間**: 5-10分

### Step 2: DRY RUN（0 epoch）
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# DRY RUN実行（初期化のみ、学習なし）
torchrun --nproc_per_node=1 train_mst.py \
  --config configs/tree_2D_guyot_dry_run.yaml \
  --device cuda \
  --cuda_visible_device 0 \
  --use_mst_train True \
  2>&1 | tee /tmp/dry_run_log.txt
```

**期待結果**:
- Config読み込み成功
- Dataset読み込み成功（Using GuyotDataset メッセージ表示）
- Model初期化成功
- Epoch 0なので学習スキップ
- エラーなく終了

**トラブルシューティング**:
- **CUDA OOM エラー**: `configs/tree_2D_guyot_dry_run.yaml`の`BATCH_SIZE: 8`を`4`または`2`に変更
- **Dataset not found**: `data/guyot_200_20_resized/`の存在確認
- **Import error**: 依存packageの再確認（`uv pip install -r requirements.txt`）

### Step 3: 1 Epoch Training
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# 1 Epoch configファイル作成
cp configs/tree_2D_guyot_test.yaml configs/tree_2D_guyot_1epoch.yaml
sed -i 's/EPOCHS: 2/EPOCHS: 1/' configs/tree_2D_guyot_1epoch.yaml
sed -i 's/test_guyot_baseline_nov14_2025/test_guyot_1epoch_nov14_2025/' configs/tree_2D_guyot_1epoch.yaml

# 1 Epoch Training実行
time torchrun --nproc_per_node=1 train_mst.py \
  --config configs/tree_2D_guyot_1epoch.yaml \
  --device cuda \
  --cuda_visible_device 0 \
  --use_mst_train True \
  2>&1 | tee /tmp/1epoch_train_log.txt
```

**期待時間**: 15-60分（GPU性能による）

**期待結果**:
- Dataset読み込み: 200 train samples
- Training開始
- Batch毎のloss表示（25 iterations予想、batch_size=8の場合）
- 1 epoch完了
- Checkpoint保存（`trained_weights/test_guyot_1epoch_nov14_2025/*.pth`）
- 正常終了

**監視項目**:
- Loss値が発散していないか（NaN, Infでないこと）
- GPU memory使用率
- 実行時間

### Step 4: Checkpoint & Loss検証（並列実行可能）

#### Step 4A: Checkpoint検証
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# 既存のテストスクリプト実行
python /tmp/test_checkpoint.py
```

**期待結果**:
- `✓ Checkpoint loaded successfully`
- Keys に`epoch`, `model_state_dict`等が含まれる

#### Step 4B: Loss解析と可視化
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# 既存のplot scriptを修正して実行
# （log fileパスを実際のものに変更）
sed -i 's|test_guyot_1epoch_nov14_2025|test_guyot_1epoch_nov14_2025|' /tmp/plot_loss.py
python /tmp/plot_loss.py
```

**期待結果**:
- `/tmp/loss_plot.png`生成
- Initial loss > Final loss（学習が進んでいる）
- Loss plotが右下がりの傾向

### Step 5: 最終レポート作成
```bash
cd /home/user/TreeFormer
source .venv/bin/activate

# レポートテンプレートから実測値を記入
# temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md を作成
```

**レポート内容**:
- Summary（成功/失敗項目）
- Dataset情報
- Training結果（iteration数、loss値、実行時間）
- 生成されたファイル一覧
- 発生した問題と対処
- Next Steps

---

## 📁 重要ファイル一覧

### 作業計画書・ログ
- `temp/WORK_PLAN_nov14_2025_training_verification.md` - 詳細な作業手順書
- `temp/PARALLEL_TASKS_nov14_2025_training_verification.md` - 並列実行可能タスク構成と作業記録
- `temp/ONBOARDING_nov14_2025.md` - プロジェクト全体のオンボーディング資料
- `temp/GPU_HANDOVER_nov14_2025.md` - **本ファイル**（GPU環境引き継ぎ資料）

### DatasetとLoader
- `guyot_dataset.py` - Guyot dataset loader（実装済み）
- `data/guyot_200_20_resized/` - Dataset本体

### Config
- `configs/tree_2D_use_mst_only1.yaml` - ベースConfig（Toulouse dataset用）
- `configs/tree_2D_guyot_test.yaml` - Guyot用Config（EPOCHS=2）
- `configs/tree_2D_guyot_dry_run.yaml` - DRY RUN用Config（EPOCHS=0）

### Training Script
- `train_mst.py` - メイン学習スクリプト（GuyotDataset対応済み）
- `models/ops/setup.py` - CUDA extension ビルドスクリプト

### テストスクリプト（`/tmp/`に配置済み）
- `/tmp/test_image_load.py` - 画像読み込みテスト
- `/tmp/test_config_load.py` - Config読み込みテスト
- `/tmp/test_dataset_load.py` - Dataset読み込みテスト
- `/tmp/test_dataloader.py` - DataLoader batch取得テスト
- `/tmp/test_checkpoint.py` - Checkpoint検証
- `/tmp/plot_loss.py` - Loss可視化

### Task結果JSON（`/tmp/`に配置済み）
- `/tmp/task_d1_dataset_load.json` - Dataset読み込み結果
- `/tmp/task_d2_dataloader.json` - DataLoader結果

---

## ⚙️ 開発環境情報

### Python環境
- **仮想環境**: `.venv/`（uv管理）
- **Python version**: 3.10+
- **有効化**: `source .venv/bin/activate`

### 主要パッケージ
```
torch>=1.7.1
torchvision>=0.8.2
numpy<2.0
opencv-python>=4.8.0
scikit-image>=0.18.0
networkx>=2.6.0
matplotlib>=3.4.0
pyyaml>=5.4.0
tqdm
setuptools
```

### Git情報
- **ブランチ**: `claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS`
- **最新コミット**: `feat: Adapt train_mst.py for Guyot dataset and CPU/GPU flexibility`
- **リモート**: origin

---

## 📝 注意事項・開発原則

### 必須ルール
1. **uv環境使用**: 全ての`pip`操作は`uv pip`で行う
2. **DRY原則**: 重複コード・設定を避ける
3. **KISS原則**: シンプルで理解しやすい実装
4. **SOLID原則**: 特にSingle Responsibility, Dependency Inversion
5. **暗黙的fallback禁止**: エラー時は明示的に処理、推測しない

### Git運用
- コミット前に必ず`git status`で変更確認
- コミットメッセージは具体的に記述（feat:/fix:/docs:プレフィックス）
- Push前に`git log --oneline -5`で履歴確認

### 作業記録の更新
- 各ステップ実行前に`date "+%Y-%m-%d %H:%M:%S %Z%z"`で時刻記録
- `temp/PARALLEL_TASKS_nov14_2025_training_verification.md`の作業記録テーブルを更新
- 問題発生時は問題発生記録テーブルに記載

---

## 🔧 トラブルシューティング

### CUDA Extension ビルド失敗
**症状**: `error: command 'gcc' failed`

**対処**:
```bash
# 必要な開発ツールインストール（rootが必要な場合はsudo使用）
apt-get update && apt-get install -y build-essential

# 再ビルド
cd /home/user/TreeFormer/models/ops
python setup.py clean
python setup.py build install
```

### CUDA OOM エラー
**症状**: `RuntimeError: CUDA out of memory`

**対処**:
```bash
# Batch sizeを削減
# configs/tree_2D_guyot_1epoch.yaml の BATCH_SIZE を変更
sed -i 's/BATCH_SIZE: 8/BATCH_SIZE: 4/' configs/tree_2D_guyot_1epoch.yaml

# または
sed -i 's/BATCH_SIZE: 8/BATCH_SIZE: 2/' configs/tree_2D_guyot_1epoch.yaml

# NUM_WORKERSも削減
sed -i 's/NUM_WORKERS: 4/NUM_WORKERS: 2/' configs/tree_2D_guyot_1epoch.yaml
```

### Loss NaN/Inf発生
**症状**: Loss値がNaN or Inf

**対処**:
```bash
# Learning rate削減
sed -i 's/LR: 1e-4/LR: 1e-5/' configs/tree_2D_guyot_1epoch.yaml

# Gradient clipping確認・調整
grep CLIP_MAX_NORM configs/tree_2D_guyot_1epoch.yaml
# → 必要に応じて値を小さく（例: 0.1 → 0.05）
```

### Dataset not found エラー
**症状**: `FileNotFoundError: data/guyot_200_20_resized/`

**対処**:
```bash
# データセット存在確認
ls -la data/guyot_200_20_resized/

# 存在しない場合、ブランチマージ必要性確認
git log --oneline --grep="Guyot" -5
```

---

## 📊 期待される成果物

### GPU環境での完了時に生成されるファイル

#### 1. Configファイル
- `configs/tree_2D_guyot_1epoch.yaml` - 1 Epoch用Config

#### 2. Checkpointファイル
- `trained_weights/test_guyot_1epoch_nov14_2025/checkpoint_*.pth`
- `trained_weights/test_guyot_1epoch_nov14_2025/model_*.pth`

#### 3. Logファイル
- `trained_weights/test_guyot_1epoch_nov14_2025/train.log`
- `/tmp/dry_run_log.txt`
- `/tmp/1epoch_train_log.txt`

#### 4. 可視化
- `/tmp/loss_plot.png` - Loss推移グラフ

#### 5. レポート
- `temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md` - 最終検証レポート

#### 6. Task結果JSON
- `/tmp/task_e4_loss_analysis.json` - Loss解析結果

---

## 🔄 作業フロー（GPU環境）

### Quick Start（推奨手順）
```bash
# 1. GPU環境確認
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. CUDA extensionビルド
cd /home/user/TreeFormer/models/ops
source ../../.venv/bin/activate
python setup.py build install

# 3. DRY RUN
cd /home/user/TreeFormer
torchrun --nproc_per_node=1 train_mst.py \
  --config configs/tree_2D_guyot_dry_run.yaml \
  --device cuda --cuda_visible_device 0 --use_mst_train True

# 4. 1 Epoch Training
cp configs/tree_2D_guyot_test.yaml configs/tree_2D_guyot_1epoch.yaml
sed -i 's/EPOCHS: 2/EPOCHS: 1/' configs/tree_2D_guyot_1epoch.yaml
sed -i 's/test_guyot_baseline/test_guyot_1epoch/' configs/tree_2D_guyot_1epoch.yaml

time torchrun --nproc_per_node=1 train_mst.py \
  --config configs/tree_2D_guyot_1epoch.yaml \
  --device cuda --cuda_visible_device 0 --use_mst_train True \
  | tee /tmp/1epoch_train_log.txt

# 5. Checkpoint & Loss検証（並列実行）
python /tmp/test_checkpoint.py &
python /tmp/plot_loss.py &
wait

# 6. レポート作成と commit & push
# （temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md作成）
git add trained_weights/ configs/ temp/
git commit -m "feat: Complete Stage 4-5 training verification on GPU"
git push -u origin claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS
```

### 想定所要時間
- CUDA extensionビルド: 5-10分
- DRY RUN: 2-5分
- 1 Epoch Training: 15-60分（GPU性能による）
- 検証・レポート作成: 10-15分
- **合計**: 約30-90分

---

## 📚 参考資料

### プロジェクト内ドキュメント
1. `temp/ONBOARDING_nov14_2025.md` - プロジェクト全体説明、セットアップ手順
2. `temp/WORK_PLAN_nov14_2025_training_verification.md` - 詳細作業手順書（全Phase）
3. `temp/PARALLEL_TASKS_nov14_2025_training_verification.md` - 並列実行タスク構成、作業記録
4. `README.md` - TreeFormerプロジェクト概要
5. `configs/tree_2D_use_mst_only1.yaml` - 元のConfig（Toulouse dataset）

### コードベース
- `guyot_dataset.py` - Guyot dataset実装詳細
- `train_mst.py` - 学習ループとdataset初期化ロジック
- `models/` - TreeFormerモデル定義
- `losses_only.py` - Loss関数定義

### 外部リソース
- TreeFormer論文（プロジェクトREADME参照）
- Guyot dataset仕様（`data/guyot_200_20_resized/`内のREADME）

---

## 💡 Tips & Best Practices

### 効率的な作業のために
1. **ログ監視**: `tail -f trained_weights/*/train.log`でリアルタイム監視
2. **GPU使用率確認**: `nvidia-smi -l 1`で継続監視
3. **Checkpoint定期保存**: Configの`VAL_INTERVAL: 1`で毎epoch保存
4. **Loss発散時の早期停止**: 最初の数iterationでNaN/Inf検出したら即座に中断

### よくあるミス
- ❌ `.venv/bin/activate`を忘れる → エラー連発
- ❌ `torchrun`を使わずに直接`python train_mst.py`実行 → RANK未設定エラー
- ❌ Batch sizeが大きすぎてOOM → 最初は小さめ（2-4）から開始推奨
- ❌ CUDA extensionビルド忘れ → ModuleNotFoundError

### デバッグテクニック
- **Dataset確認**: `python /tmp/test_dataset_load.py`で即座に確認
- **Config確認**: `python /tmp/test_config_load.py configs/xxx.yaml`
- **Import確認**: `python -c "import MultiScaleDeformableAttention"`
- **GPU確認**: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

---

## 🎓 継承すべき開発思想

### CPU環境エージェントから学んだこと
1. **アトミック性**: 1ステップ = 1操作、分割不可
2. **順序性**: 依存関係を明示、順番厳守
3. **検証可能性**: 各ステップに成功条件を明記
4. **完結性**: 各ステップは独立して完結可能
5. **明示的エラー処理**: 暗黙的fallbackは禁止、全てのエラーを明示的に処理

### 記録の重要性
- 全ての操作に時刻記録（`date`コマンド）
- 成功/失敗を明確に記録
- 問題と対処を必ず記載
- 次の作業者が追跡可能な粒度で記録

---

## 📞 質問・問題発生時の対応

### 問題が発生した場合
1. **エラーメッセージ全文をコピー**（`2>&1 | tee error.log`活用）
2. **temp/PARALLEL_TASKS_nov14_2025_training_verification.md**の問題発生記録テーブルに記載
3. **対処手順**:
   - エラーメッセージでGoogle検索
   - 本ドキュメントのトラブルシューティング参照
   - `temp/WORK_PLAN_nov14_2025_training_verification.md`の該当Phase確認
4. **解決したら**:
   - 対処内容を問題発生記録テーブルに追記
   - 作業記録テーブルを更新

### 不明点がある場合
- `temp/ONBOARDING_nov14_2025.md`を再確認
- `temp/WORK_PLAN_nov14_2025_training_verification.md`の詳細手順を参照
- コード内のコメントを確認（特に`guyot_dataset.py`, `train_mst.py`）

---

## ✅ 引き継ぎチェックリスト

### GPU環境エージェントが作業開始前に確認すべき項目
- [ ] 本ドキュメント全体を熟読した
- [ ] GPU利用可能を確認した（`torch.cuda.is_available() == True`）
- [ ] uv環境が正しく設定されている（`which python`が`.venv/`を指す）
- [ ] ブランチが正しい（`claude/prepare-temp-docs-01D32vfvA14H5qmVAMvjgmhS`）
- [ ] データセットが存在する（`ls data/guyot_200_20_resized/`）
- [ ] 作業記録ファイルの場所を把握した（`temp/PARALLEL_TASKS_nov14_2025_training_verification.md`）
- [ ] 開発原則（DRY/KISS/SOLID、暗黙的fallback禁止）を理解した

### 作業完了時に確認すべき項目
- [ ] Stage 4-5の全タスクが完了（チェックリスト全て[x]）
- [ ] Checkpointが生成され、読み込み可能
- [ ] Lossが収束傾向
- [ ] 最終レポートが作成済み（`temp/TRAINING_VERIFICATION_REPORT_nov14_2025.md`）
- [ ] 全ての変更をcommit & push済み
- [ ] 作業記録が完全に更新済み

---

## 🏁 最終目標再確認

### 成功の定義
GPU環境エージェントは、以下を達成すれば**作業完了**とする：

1. ✅ CUDA extensionビルド成功
2. ✅ DRY RUN（0 epoch）エラーなく完了
3. ✅ 1 Epoch Training正常完了
4. ✅ Checkpoint生成・検証成功
5. ✅ Loss収束確認（Initial > Final）
6. ✅ 最終レポート作成完了
7. ✅ 全変更commit & push完了

---

## 🙏 引き継ぎメッセージ

CPU環境エージェントより：

> Stage 0-3（事前調査〜Dataset Loader検証）は完璧に完了しています。
> GuyotDatasetは完全に動作し、train_mst.pyもGuyotに対応済みです。
> あとはGPU環境でCUDA extensionをビルドし、学習を実行するだけです。
>
> 全ての準備は整っています。頑張ってください！
>
> 問題が発生したら、このドキュメントと`temp/WORK_PLAN_nov14_2025_training_verification.md`を参照してください。
> 必ず成功します。

---

**作成者**: CPU環境エージェント
**作成日時**: 2025-11-14 10:56:01 UTC+0000
**ドキュメントバージョン**: 1.0
**次回更新**: GPU環境エージェントによる作業完了時

---

**END OF GPU HANDOVER DOCUMENT**
