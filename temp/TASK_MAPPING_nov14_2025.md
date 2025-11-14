# Task Mapping: Sequential WORK_PLAN → Parallel TASKS
## 作業書対応関係マッピング

**作成日**: 2025-11-14
**目的**: 元の順次実行WORK_PLANと並列実行PARALLEL_TASKSの対応関係を明確化

---

## 📊 概要

### 変更のポイント
- **元**: 6 Phase、105 Steps（順次実行想定）
- **新**: 6 Stage、15 Tasks（並列実行最適化）
- **削減効果**: Stage 0-1で約50%の時間短縮見込み

### 並列化の方針
1. **事前調査系**: 相互依存なし → 完全並列化
2. **テストスクリプト作成**: 読み込み系のみ → 完全並列化
3. **セットアップ系**: Config作成とpackage installは並列化
4. **実行系**: 依存関係が強い → 順次実行

---

## Stage 0: 事前調査フェーズ

### Task A1: Dataset実ファイル確認
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| A1 全体 | Phase 1, Task 1.1 | Dataset実ファイル確認 |
| A1-Step1 | Step 1.1.1 | Train/Validationセット確認 |
| A1-Step2 | Step 1.1.2 | 画像数カウント（Train） |
| A1-Step3 | Step 1.1.3 | アノテーション数カウント（Train） |
| A1-Step4 | Step 1.1.4-1.1.5 | Testセット確認・カウント |
| A1-Step5 | Step 1.1.6 | サンプル画像サイズ確認 |
| A1-Output | - | JSON出力（新規追加） |

**統合内容**: Step 1.1.7-1.1.10（画像読み込みテスト）は Task B1へ移動

---

### Task A2: 既存Config全パラメータ解析
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| A2 全体 | Phase 1, Task 1.2 | 既存Config全パラメータ解析 |
| A2-Step1 | Step 1.2.1 | Config読み込み |
| A2-Step2 | Step 1.2.2 | Dataset関連パラメータ抽出 |
| A2-Step3 | Step 1.2.3 | Model関連パラメータ抽出 |
| A2-Step4 | Step 1.2.4 | Training関連パラメータ抽出 |
| A2-Step5 | Step 1.2.5-1.2.6 | Config読み込みテストスクリプト実行 |
| A2-Output | - | JSON出力（新規追加、Guyot変更必要箇所含む） |

**統合内容**: テストスクリプト作成部分は Task B2へ移動

---

### Task A3: train_mst.pyコード解析
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| A3 全体 | Phase 2, Task 2.1 | Dataset loaderコード解析 |
| A3-Step1 | Step 2.1.1 | Import文確認 |
| A3-Step2 | Step 2.1.2 | Dataset初期化部分特定 |
| A3-Step3 | Step 2.1.3 | Dataset初期化コード詳細確認 |
| A3-Step4 | Step 2.1.4 | データセットクラスファイル検索 |
| A3-Step5 | Step 2.1.5 | Guyot dataset class検索 |
| A3-DataLoader | - | DataLoader箇所特定（新規追加） |
| A3-Argparse | - | Argument parser確認（新規追加） |
| A3-Output | - | JSON出力（dataset class名特定） |

**統合内容**: Task 2.2（Guyot dataset class存在確認）も含む

---

### Task A4: 依存関係確認
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| A4 全体 | Phase 3, Task 3.1-3.2 | requirements/現在環境確認 |
| A4-Step1 | Step 3.1.1-3.1.2 | requirements/pyproject.toml確認 |
| A4-Step2 | Step 3.2.1-3.2.2 | 仮想環境・インストール済みpackage確認 |
| A4-Step3 | Step 3.2.3 | 主要package確認 |
| A4-Step4 | Step 3.1.4 | train_mst.pyから必要package抽出 |
| A4-Step5 | - | Package名正規化（新規追加） |
| A4-Output | - | JSON出力（不足package特定） |

**統合内容**: Task 3.3（不足package特定）も含む、Step 3.2.4（importテスト）は Task B4へ移動

---

## Stage 1: テストスクリプト作成

### Task B1: 画像読み込みテストスクリプト作成
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| B1 全体 | Step 1.1.7 | 画像読み込みテストスクリプト作成 |
| B1-Script | Step 1.1.7 | /tmp/test_image_load.py作成 |
| B1-Verify | Step 1.1.8 | 実行検証（成果物確認） |

**変更内容**: スクリプト作成をTask化、エラー処理（Step 1.1.9-1.1.10）は実行時対処

---

### Task B2: Config読み込みテストスクリプト作成
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| B2 全体 | Step 1.2.5 | Config読み込みテストスクリプト作成 |
| B2-Script | Step 1.2.5 | /tmp/test_config_load.py作成 |
| B2-Verify | Step 1.2.6 | 実行検証 |

---

### Task B3: Dataset importテストスクリプト作成
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| B3 全体 | Step 2.2.4 | Dataset importテストスクリプト作成 |
| B3-Script | Step 2.2.4 | /tmp/test_dataset_import.py作成 |
| B3-Verify | Step 2.2.5 | 実行検証 |

**統合内容**: Task A3の結果（dataset class候補）を反映

---

### Task B4: 依存packageインポートテストスクリプト作成
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| B4 全体 | Step 3.2.4 | 主要packageインポートテスト |
| B4-Script | Step 3.2.4 | /tmp/test_imports.py作成 |
| B4-Verify | 実行検証 | 全packageインポート成功確認 |

---

## Stage 2: Config・依存関係セットアップ

### Task C1: Guyot用Config作成
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| C1 全体 | Phase 1, Task 1.3 | Guyot用Config作成 |
| C1-Step1 | Step 1.3.1 | ベースConfigコピー |
| C1-Step2 | Step 1.3.2 | DATA_PATH変更 |
| C1-Step3 | Step 1.3.3 | DATASET名変更 |
| C1-Step4 | Step 1.3.4 | EPOCHS変更（テスト用） |
| C1-Step5 | Step 1.3.5 | exp_name変更 |
| C1-Verify | Step 1.3.6-1.3.7 | 作成Config確認・読み込みテスト |

**並列実行**: Task C3（package install）と並列実行可能

---

### Task C2: Config差分検証
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| C2 全体 | Phase 1, Task 1.4 | Config差分確認テスト |
| C2-Diff | Step 1.4.1 | 差分表示 |
| C2-Script | Step 1.4.2 | Validation script作成 |
| C2-Verify | Step 1.4.3 | Validation実行 |

**依存**: Task C1完了後に実行

---

### Task C3: 依存package不足分インストール
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| C3 全体 | Phase 3, Task 3.4 | 追加packageインストール |
| C3-Step1 | Task A4結果 | 不足package特定（A4の出力から） |
| C3-Step2 | Step 3.4.1 | 一括インストール |
| C3-Verify | Step 3.4.2-3.4.3 | インストール確認・再importテスト |

**並列実行**: Task C1と並列実行可能（相互依存なし）

**エラー処理**: Step 3.4.4-3.4.5は実行時対処

---

## Stage 3: Dataset Loader検証

### Task D1: Dataset読み込みテスト
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| D1 全体 | Phase 2, Task 2.3 | Dataset読み込みテスト実行 |
| D1-Step1 | Step 2.3.1 | Minimal dataset load script作成 |
| D1-Step2 | Step 2.3.2 | train_mst.pyのdataset初期化コード抽出 |
| D1-Step3 | Step 2.3.3 | 抽出コードを統合 |
| D1-Execute | Step 2.3.4 | Dataset読み込みテスト実行 |

**統合内容**: Task A3の結果（dataset class）を利用

**エラー処理**: Step 2.3.5-2.3.6は実行時対処

---

### Task D2: DataLoader batch取得テスト
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| D2 全体 | Phase 2, Task 2.4 | Batch取得テスト |
| D2-Step1 | Step 2.4.1 | DataLoader作成テストスクリプト |
| D2-Step2 | Step 2.4.2 | train_mst.pyのDataLoader部分抽出 |
| D2-Step3 | Step 2.4.3 | DataLoaderコード統合 |
| D2-Execute | Step 2.4.4 | DataLoader実行 |
| D2-Verify | Step 2.4.5 | Batch内容妥当性確認 |

**エラー処理**: Step 2.4.6（Shape不一致）は実行時対処

---

## Stage 4: Training実行

### Task E1: DRY RUN (0 epoch)
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| E1 全体 | Phase 4, Task 4.2 | DRY RUN実行 |
| E1-Step1 | Step 4.2.1 | Config一時変更（EPOCHS=0） |
| E1-Verify | Step 4.2.2 | 変更確認 |
| E1-Execute | Step 4.2.3 | DRY RUN実行 |
| E1-Log | Step 4.2.4 | 出力ログ確認 |

**統合内容**: Task 4.1（引数確認）も含む

**エラー処理**: Step 4.2.5-4.2.6は実行時対処

---

### Task E2: 1 Epoch Training実行
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| E2 全体 | Phase 4, Task 4.3 | 1 Epoch学習実行 |
| E2-Step1 | Step 4.3.1 | 1 Epoch config作成 |
| E2-Verify | Step 4.3.2 | Config確認 |
| E2-Execute | Step 4.3.3 | 1 Epoch学習実行 |
| E2-Monitor | Step 4.3.4 | 実行中ログ監視 |

---

### Task E3: Checkpoint検証
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| E3 全体 | Phase 4, Task 4.4 | Checkpoint保存確認 |
| E3-Step1 | Step 4.4.1 | 保存先ディレクトリ確認 |
| E3-Step2 | Step 4.4.2 | Checkpoint内容確認 |
| E3-Test | Step 4.4.3 | Checkpoint読み込みテスト |

**並列実行**: Task E4と並列実行可能（両方Task E2完了後）

**エラー処理**: Step 4.4.4は実行時対処

---

### Task E4: Loss解析と可視化
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| E4 全体 | Phase 4, Task 4.5 | Log出力確認 |
| E4-Step1 | Step 4.5.1 | Log file内容表示 |
| E4-Step2 | Step 4.5.2 | Loss値抽出 |
| E4-Plot | Step 4.5.3 | Loss推移可視化 |
| E4-Verify | Step 4.5.4 | Plot画像確認 |
| E4-Output | - | JSON出力（新規追加） |

**並列実行**: Task E3と並列実行可能

---

## Stage 5: レポート作成

### Task F1: Training検証レポート作成
| Parallel Task | 元WORK_PLAN | 説明 |
|--------------|------------|------|
| F1 全体 | Phase 6, Task 6.1-6.4 | 成功基準検証・レポート作成 |
| F1-Training完走 | Task 6.1 | Training完走確認 |
| F1-Checkpoint | Task 6.2 | Checkpoint生成確認 |
| F1-Loss収束 | Task 6.3 | Loss収束傾向確認 |
| F1-Report | Task 6.4 | 最終レポート作成 |
| F1-集約 | - | 全Task結果JSON集約（新規追加） |

**統合内容**: Phase 6全体を1 Taskに統合、並列実行結果も含む

---

## Phase 5: エラートラブルシューティング

### マッピング方針
| 元WORK_PLAN | Parallel Tasks | 説明 |
|------------|---------------|------|
| Phase 5全体 | 各Task内 | エラー処理として各Task内に組み込み |
| Task 5.1 | Task D1, D2内 | Dataset関連エラー対処 |
| Task 5.2 | Task E1, E2内 | Model関連エラー対処 |
| Task 5.3 | Task E1, E2内 | Memory関連エラー対処 |
| Task 5.4 | Task E4内 | Loss関連エラー対処 |

**変更内容**: 独立Phaseではなく、各実行Task内のエラー処理として統合

---

## 📊 Step数比較

### 元WORK_PLAN
| Phase | Task数 | Step数 |
|-------|-------|-------|
| Phase 1 | 4 | 14 |
| Phase 2 | 4 | 24 |
| Phase 3 | 4 | 17 |
| Phase 4 | 5 | 24 |
| Phase 5 | 4 | 12 |
| Phase 6 | 4 | 14 |
| **合計** | **25** | **105** |

### Parallel TASKS
| Stage | Task数 | 並列度 | 想定時間 |
|-------|-------|-------|---------|
| Stage 0 | 4 | 完全並列 | 5-7分 |
| Stage 1 | 4 | 完全並列 | 2-3分 |
| Stage 2 | 3 | 一部並列（C1∥C3） | 5-10分 |
| Stage 3 | 2 | 順次 | 10-12分 |
| Stage 4 | 4 | 順次→並列（E3∥E4） | 20-35分 |
| Stage 5 | 1 | 順次 | 5-7分 |
| **合計** | **18** (-28%) | - | **47-74分** |

**削減効果**:
- Task数: 25 → 18 (-28%)
- 並列実行により実時間: 約50-60分（順次実行の場合90-120分想定）

---

## 🔄 並列実行フロー

### 時系列実行イメージ
```
時刻   0min   5min   10min  15min  20min  25min  30min  35min  40min  45min  50min  55min
       |------|------|------|------|------|------|------|------|------|------|------|
Stage0 [A1A2A3A4]  完了
Stage1       [B1B2B3B4] 完了
Stage2             [C1      ] 完了
                   [C3      ] 完了（並列）
                          [C2] 完了
Stage3                       [D1    ] 完了
                                  [D2  ] 完了
Stage4                                  [E1] 完了
                                         [E2          ] 完了
                                                   [E3E4] 完了（並列）
Stage5                                                     [F1    ] 完了
```

**順次実行の場合**: A1→A2→A3→A4→B1→... = 90-120分
**並列実行の場合**: max(A1,A2,A3,A4)→max(B1,B2,B3,B4)→... = 47-74分

---

## 📝 実行推奨順序

### 1回目実行（検証）
1. **Stage 0のみ並列実行**: 環境確認
2. **Stage 1のみ並列実行**: テストスクリプト検証
3. **Stage 2順次実行**: Config作成確認
4. **Stage 3-5順次実行**: 実際のTraining検証

### 2回目以降（高速化）
1. **Stage 0-1完全並列**: 5 subagent同時起動
2. **Stage 2一部並列**: 2 subagent同時起動
3. **Stage 3-5順次**: 必要に応じて

---

## 🎯 成果物対応表

| 成果物 | 元WORK_PLAN | Parallel TASKS |
|-------|------------|---------------|
| dataset確認結果 | Phase 1, Task 1.1 | `/tmp/task_a1_dataset_check.json` |
| config解析結果 | Phase 1, Task 1.2 | `/tmp/task_a2_config_analysis.json` |
| train_mst解析 | Phase 2, Task 2.1 | `/tmp/task_a3_train_mst_analysis.json` |
| 依存関係確認 | Phase 3, Task 3.1-3.3 | `/tmp/task_a4_dependency_check.json` |
| Guyot Config | Phase 1, Task 1.3 | `configs/tree_2D_guyot_test.yaml` |
| 1 Epoch学習 | Phase 4, Task 4.3 | `trained_weights/test_guyot_1epoch_*/` |
| Loss解析 | Phase 4, Task 4.5 | `/tmp/task_e4_loss_analysis.json` |
| 検証レポート | Phase 6, Task 6.4 | `temp/PARALLEL_TRAINING_VERIFICATION_REPORT_nov14_2025.md` |

---

## 🔧 実装詳細の変更点

### 1. JSON出力の追加
- **目的**: Task間のデータ受け渡しを構造化
- **例**: Task A1 → Task C1（dataset確認結果をConfig作成時に参照）

### 2. エラー処理の統合
- **元**: Phase 5として独立
- **新**: 各Task内の「エラー処理」セクションに統合
- **利点**: Task実行時に即座に対処可能

### 3. テストスクリプトの分離
- **元**: 実行と作成が混在
- **新**: Stage 1でスクリプト作成、後続Stageで実行
- **利点**: スクリプト並列作成可能

### 4. 並列実行用メタデータ
- **依存関係**: 各Task冒頭に明記
- **想定時間**: 並列実行計画立案に使用
- **実行モデル**: haiku（軽量）/sonnet（複雑）の使い分け

---

**END OF TASK MAPPING DOCUMENT**
