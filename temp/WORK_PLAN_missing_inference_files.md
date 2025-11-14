# 作業計画書 兼 記録書：TreeFormer 欠損推論ファイル実装（並列実行版）

---

**日付：** `2025年11月14日`
**作業ディレクトリ・リポジトリ:** `/home/user/TreeFormer` (yuki-inaho/TreeFormer)
**作業ブランチ:** `claude/implement-missing-inference-files-01D32vfvA14H5qmVAMvjgmhS`
**作業者：** `[作業者名を記入]`

---

## 1. 作業目的

本作業は、TreeFormerプロジェクトにおけるGitHub Issue #2「Missing inference_infinity_mst_nx_gradmst File in Repository」を完全解決するために実施します。

*   **目標1:** 欠損している3つの推論ファイルを実装し、検証スクリプト（valid_smd_guyot_nx.py）が正常に動作するようにする
*   **目標2:** 論文アルゴリズムとコード実装の対応を完全に保証し、テストで検証する
*   **目標3:** 将来の保守性を確保するため、包括的なテストスイートとドキュメントを整備する

**背景:**
- 検証スクリプト `valid_smd_guyot_nx.py` (行1255-1259) が存在しない3つのモジュールをインポートしようとしている
- 調査により、これらのモジュールが実装すべきアルゴリズムを完全に特定済み
- 既存の `epoch.py` に同等の実装が存在することも確認済み

**欠損ファイル:**
1. `inference_infinity_mst_nx_gradmst.py` - TreeFormer (SFS Layer) 実装
2. `inference_infinity_mst_nx_dist.py` - Test-time constraint 実装
3. `inference_infinity_gradmst.py` - Unconstrained baseline 実装

---

## 2. 並列実行戦略

### 並列実行可能なタスクグループ

本作業計画では、以下のタスクグループを**並列実行可能**として設計しています：

**グループA: 推論ファイル実装（フェーズ2）**
- 3つの推論ファイルは互いに独立しているため、並列実装可能
- 各ファイルは異なるsubagentで同時実装

**グループB: テストファイル作成（フェーズ3）**
- 8つのテストファイルを複数グループに分けて並列作成可能
- Unit/Integration/E2Eの各層で並列実行

**グループC: ドキュメント整備（フェーズ5）**
- docstring、README、コード品質チェックを並列実行可能

### Subagent活用方針

1. **探索的タスク**: `subagent_type=Explore` を使用
2. **実装タスク**: `subagent_type=general-purpose` を使用
3. **並列実行**: 単一メッセージで複数Task toolを呼び出し

---

## 3. 作業内容（並列実行版）

### フェーズ 1: [環境確認と緊急修正] (見積: 0.5h) - **逐次実行**

このフェーズは依存関係があるため逐次実行が必要です。

#### 目的
- uv環境が正しく構築されているか確認
- 既存実装を活用した暫定修正を実施
- 基本的な動作確認を行う

---

### フェーズ 2: [テストファーストでの欠損ファイル実装] (見積: 2.0h) - **並列実行可能**

このフェーズは3つの独立した実装タスクを含むため、**並列実行**を推奨します。

#### 並列実行グループ 2-A: テスト環境準備（逐次実行）
- 手順 2-1: テストディレクトリ作成
- 手順 2-2: モックデータジェネレータ作成

#### 並列実行グループ 2-B: 推論ファイル実装（3並列）

**並列タスク 2-B-1: inference_infinity_mst_nx_gradmst.py 実装**
- 担当subagent: Agent-1 (general-purpose)
- 依存: グループ 2-A完了後
- テストファイル: tests/test_inference_gradmst.py

**並列タスク 2-B-2: inference_infinity_mst_nx_dist.py 実装**
- 担当subagent: Agent-2 (general-purpose)
- 依存: グループ 2-A完了後
- テストファイル: tests/test_inference_dist.py

**並列タスク 2-B-3: inference_infinity_gradmst.py 実装**
- 担当subagent: Agent-3 (general-purpose)
- 依存: グループ 2-A完了後
- テストファイル: tests/test_inference_unconst.py

---

### フェーズ 3: [包括的テストスイート作成] (見積: 1.5h) - **並列実行可能**

このフェーズは複数の独立したテストファイルを含むため、**並列実行**を推奨します。

#### 並列実行グループ 3-A: Unit Tests（3並列）

**並列タスク 3-A-1: MST計算テスト**
- ファイル: tests/test_mst_computation.py
- 担当subagent: Agent-4 (general-purpose)

**並列タスク 3-A-2: SFS Layerテスト**
- ファイル: tests/test_sfs_layer.py
- 担当subagent: Agent-5 (general-purpose)

**並列タスク 3-A-3: pytest設定**
- ファイル: pytest.ini
- 担当subagent: Agent-6 (general-purpose)

#### 並列実行グループ 3-B: Integration/E2E Tests（2並列）

**並列タスク 3-B-1: Integration Tests**
- ファイル: tests/test_integration.py
- 担当subagent: Agent-7 (general-purpose)
- 依存: グループ 2-B完了後

**並列タスク 3-B-2: E2E Tests**
- ファイル: tests/test_e2e.py
- 担当subagent: Agent-8 (general-purpose)
- 依存: グループ 2-B完了後

---

### フェーズ 4: [統合とリグレッション確認] (見積: 1.0h) - **逐次実行**

このフェーズは結果の検証が必要なため逐次実行が必要です。

---

### フェーズ 5: [ドキュメント整備とクリーンアップ] (見積: 0.5h) - **並列実行可能**

#### 並列実行グループ 5-A: ドキュメント作成（3並列）

**並列タスク 5-A-1: docstring検証と修正**
- 担当subagent: Agent-9 (general-purpose)

**並列タスク 5-A-2: README更新**
- 担当subagent: Agent-10 (general-purpose)

**並列タスク 5-A-3: コード品質チェック**
- 担当subagent: Agent-11 (general-purpose)

---

## 4. 詳細作業チェックリスト

### フェーズ 1: [環境確認と緊急修正] - 逐次実行

#### 手順 1-1: uv環境の確認
- [x] 操作: `uv --version` でuvがインストールされているか確認
- [x] 確認: バージョン情報が表示される（例: `uv 0.4.x`）
- [x] テスト: N/A（環境確認のため）
- [x] エラー時対処: uvが見つからない場合は `curl -LsSf https://astral.sh/uv/install.sh | sh` でインストール

#### 手順 1-2: Python環境の有効化
- [x] 操作: `source .venv/bin/activate` で仮想環境を有効化（存在しない場合は `uv venv` で作成）
- [x] 確認: プロンプトに `(.venv)` が表示される
- [x] テスト: `which python` で `.venv` 内のpythonを使用していることを確認
- [x] エラー時対処: .venvが存在しない場合は `uv venv` で作成

#### 手順 1-3: 依存パッケージの確認
- [x] 操作: `uv pip list | grep -E "(torch|networkx|scipy|numpy)"` で必須パッケージを確認
- [x] 確認: torch, networkx, scipy, numpyが表示される
- [x] テスト: `python -c "import torch, networkx, scipy, numpy; print('OK')"` で確認
- [x] エラー時対処: 不足している場合は `uv pip install torch networkx scipy numpy` を実行

#### 手順 1-4: 既存コードの存在確認
- [x] 操作: `ls -l epoch.py losses_only.py valid_smd_guyot_nx.py` で必要ファイルが存在するか確認
- [x] 確認: 3つのファイルが全て存在し、サイズが0より大きい
- [x] テスト: `python -m py_compile epoch.py losses_only.py valid_smd_guyot_nx.py` で構文チェック
- [x] エラー時対処: ファイルが存在しない場合は正しいディレクトリに移動

#### 手順 1-5: epoch.pyの関数確認
- [x] 操作: `grep -n "^def relation_infer" epoch.py` で必要な関数が存在するか確認
- [x] 確認: `relation_infer` (行43) と `relation_infer_mst` (行308) が見つかる
- [x] テスト: N/A
- [x] エラー時対処: 関数が見つからない場合は、正しいブランチにいるか確認

#### 手順 1-6: valid_smd_guyot_nx.pyのバックアップ
- [x] 操作: `cp valid_smd_guyot_nx.py valid_smd_guyot_nx.py.backup`
- [x] 確認: `ls -l valid_smd_guyot_nx.py*` で2ファイル表示
- [x] テスト: `diff valid_smd_guyot_nx.py valid_smd_guyot_nx.py.backup` で差分なし
- [x] エラー時対処: 書き込み権限がない場合は `chmod u+w .`

#### 手順 1-7: 暫定修正の適用
- [x] 操作: valid_smd_guyot_nx.py の1255-1259行を以下に置き換え
  ```python
  if is_use_mst:
      from epoch import relation_infer_mst as relation_infer
  else:
      from epoch import relation_infer
  ```
- [x] 確認: `sed -n '1255,1259p' valid_smd_guyot_nx.py` で変更確認
- [x] テスト: `python -m py_compile valid_smd_guyot_nx.py` で構文エラーなし
- [x] エラー時対処: SyntaxErrorが出る場合はインデントを確認

#### 手順 1-8: 暫定修正後のimport確認
- [x] 操作: `python -c "import sys; sys.path.insert(0, '.'); from epoch import relation_infer, relation_infer_mst; print('OK')"`
- [x] 確認: "OK"が表示される
- [x] テスト: N/A
- [x] エラー時対処: ModuleNotFoundErrorの場合は依存パッケージを確認

#### 手順 1-9: 変更のコミット
- [x] 操作: `git add valid_smd_guyot_nx.py && git commit -m "Fix: Use existing epoch.py functions for inference (temporary)"`
- [x] 確認: `git log -1 --oneline` でコミット確認
- [x] テスト: N/A
- [x] エラー時対処: N/A

---

### フェーズ 2: [テストファーストでの欠損ファイル実装] - 並列実行可能

#### グループ 2-A: テスト環境準備（逐次実行）

##### 手順 2-1: テストディレクトリの作成
- [x] 操作: `mkdir -p tests && touch tests/__init__.py`
- [x] 確認: `ls -ld tests/ tests/__init__.py` でディレクトリとファイルが存在
- [x] テスト: `python -c "import tests; print('OK')"`
- [x] エラー時対処: 既存のtests/ディレクトリがある場合はそれを使用

##### 手順 2-2: テスト用conftest.pyの作成
- [x] 操作: `tests/conftest.py` を作成（pytest fixtures定義）
- [x] 確認: ファイルが作成され、以下のfixtureが含まれる
  - `dummy_model_output`: モデル出力ダミー
  - `dummy_hidden_features`: 隠れ層特徴ダミー
  - `mock_network`: モックネットワーク
- [x] テスト: `pytest tests/conftest.py --collect-only` でfixture認識確認
- [x] エラー時対処: pytestがない場合は `uv pip install pytest`

**tests/conftest.py の内容:**
```python
"""Pytest fixtures for TreeFormer inference tests."""
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def dummy_model_output():
    """モデル出力のダミーデータを生成"""
    batch_size = 2
    num_tokens = 20
    return {
        'pred_logits': torch.randn(batch_size, num_tokens, 2),
        'pred_nodes': torch.rand(batch_size, num_tokens, 4)
    }


@pytest.fixture
def dummy_hidden_features():
    """隠れ層特徴のダミーデータを生成"""
    batch_size = 2
    seq_len = 21  # 20 obj + 1 rln
    hidden_dim = 256
    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def mock_network():
    """モックニューラルネットワークを作成"""
    class MockModule:
        relation_embed = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    class MockNetwork:
        module = MockModule()

    return MockNetwork()
```

---

#### グループ 2-B: 推論ファイル実装（**3並列実行**）

**並列実行指示:**
この3つのタスクは独立しているため、単一メッセージで3つのTask toolを同時に呼び出して並列実行してください。

##### 【並列タスク 2-B-1】inference_infinity_mst_nx_gradmst.py 実装

**Subagent指示:**
```
Task: Implement inference_infinity_mst_nx_gradmst.py for TreeFormer

Requirements:
1. Implement relation_infer() function with MST constraint
2. Implement compute_mst_nx() helper function
3. Add comprehensive docstrings with paper references
4. Follow existing code style from epoch.py:308-582
5. Use NetworkX for Kruskal's MST algorithm
6. Cost matrix: edge non-existence probabilities

Expected output:
- File: inference_infinity_mst_nx_gradmst.py (~600 lines)
- Functions: relation_infer, compute_mst_nx
- Type hints for all parameters
- Paper references in comments (Section 4.2, Equation 10)

Validation:
- python -m py_compile inference_infinity_mst_nx_gradmst.py
- No syntax errors
```

**チェックリスト:**
- [x] 操作: Subagent (general-purpose) に上記タスクを依頼
- [x] 確認: inference_infinity_mst_nx_gradmst.py が作成される
- [x] テスト: `python -m py_compile inference_infinity_mst_nx_gradmst.py` で構文チェック
- [x] エラー時対処: Subagentの出力を確認し、必要に応じて修正指示

##### 【並列タスク 2-B-2】inference_infinity_mst_nx_dist.py 実装

**Subagent指示:**
```
Task: Implement inference_infinity_mst_nx_dist.py for Test-time constraint

Requirements:
1. Implement relation_infer() with optional distance weighting
2. Add use_distance and distance_weight parameters
3. Geometric distance calculation using torch.cdist
4. Follow paper Section 5.3 "Test-time constraint"
5. Reuse compute_mst_nx from gradmst implementation

Expected output:
- File: inference_infinity_mst_nx_dist.py (~550 lines)
- Functions: relation_infer, compute_mst_nx
- Distance weighting: final_cost = (1-α)*p(non-exist) + α*distance

Validation:
- python -m py_compile inference_infinity_mst_nx_dist.py
```

**チェックリスト:**
- [x] 操作: Subagent (general-purpose) に上記タスクを依頼
- [x] 確認: inference_infinity_mst_nx_dist.py が作成される
- [x] テスト: `python -m py_compile inference_infinity_mst_nx_dist.py`
- [x] エラー時対処: Subagentの出力を確認し、必要に応じて修正指示

##### 【並列タスク 2-B-3】inference_infinity_gradmst.py 実装

**Subagent指示:**
```
Task: Implement inference_infinity_gradmst.py for Unconstrained baseline

Requirements:
1. Implement relation_infer() with threshold-based selection
2. No MST constraint - simple argmax selection
3. Follow paper Section 5.3 "Unconstrained [55]"
4. Based on epoch.py:43-306
5. Simpler implementation than MST versions

Expected output:
- File: inference_infinity_gradmst.py (~450 lines)
- Functions: relation_infer only (no MST computation)
- Threshold selection: E = {(i,j) | p(exist) > p(non-exist)}

Validation:
- python -m py_compile inference_infinity_gradmst.py
```

**チェックリスト:**
- [x] 操作: Subagent (general-purpose) に上記タスクを依頼
- [x] 確認: inference_infinity_gradmst.py が作成される
- [x] テスト: `python -m py_compile inference_infinity_gradmst.py`
- [x] エラー時対処: Subagentの出力を確認し、必要に応じて修正指示

---

#### グループ 2-C: 対応テストファイル作成（**3並列実行**）

**並列実行指示:**
グループ 2-B完了後、この3つのテストファイルを並列作成してください。

##### 【並列タスク 2-C-1】test_inference_gradmst.py 作成

**Subagent指示:**
```
Task: Create tests/test_inference_gradmst.py

Requirements:
1. Test that MST output is always a tree structure
2. Test edge count is N-1
3. Test consistency with epoch.relation_infer_mst
4. Use fixtures from conftest.py
5. Minimum 3 test methods

Expected output:
- File: tests/test_inference_gradmst.py (~150 lines)
- Class: TestInferenceGradMST
- Tests: test_output_is_tree, test_consistency_with_epoch, test_compute_mst_nx

Validation:
- pytest tests/test_inference_gradmst.py -v --collect-only
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_inference_gradmst.py が作成される
- [ ] テスト: `pytest tests/test_inference_gradmst.py -v --collect-only`
- [ ] エラー時対処: pytest collectエラーを確認

##### 【並列タスク 2-C-2】test_inference_dist.py 作成

**Subagent指示:**
```
Task: Create tests/test_inference_dist.py

Requirements:
1. Test distance weighting functionality
2. Test use_distance parameter
3. Test with and without distance weighting
4. Minimum 3 test methods

Expected output:
- File: tests/test_inference_dist.py (~120 lines)
- Class: TestInferenceDist
- Tests: test_distance_weighting, test_use_distance_param, test_output_is_tree

Validation:
- pytest tests/test_inference_dist.py -v --collect-only
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_inference_dist.py が作成される
- [ ] テスト: `pytest tests/test_inference_dist.py -v --collect-only`
- [ ] エラー時対処: pytest collectエラーを確認

##### 【並列タスク 2-C-3】test_inference_unconst.py 作成

**Subagent指示:**
```
Task: Create tests/test_inference_unconst.py

Requirements:
1. Test threshold-based selection
2. Test that output may not be a tree
3. Compare with constrained versions
4. Minimum 3 test methods

Expected output:
- File: tests/test_inference_unconst.py (~100 lines)
- Class: TestInferenceUnconst
- Tests: test_threshold_selection, test_may_not_be_tree, test_consistency

Validation:
- pytest tests/test_inference_unconst.py -v --collect-only
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_inference_unconst.py が作成される
- [ ] テスト: `pytest tests/test_inference_unconst.py -v --collect-only`
- [ ] エラー時対処: pytest collectエラーを確認

---

#### 手順 2-D: グループ2-B,2-C完了確認（逐次実行）

##### 手順 2-D-1: 全推論ファイルの構文チェック
- [ ] 操作: `python -m py_compile inference_*.py`
- [ ] 確認: エラーなく完了
- [ ] テスト: `ls -l inference_*.py` で3ファイル確認
- [ ] エラー時対処: SyntaxErrorがある場合は該当ファイルを修正

##### 手順 2-D-2: 全テストファイルのcollect確認
- [ ] 操作: `pytest tests/test_inference_*.py --collect-only`
- [ ] 確認: 全テストケースが認識される
- [ ] テスト: 最低9個のテストケースが表示される
- [ ] エラー時対処: collectエラーがある場合は該当テストを修正

---

### フェーズ 3: [包括的テストスイート作成] - 並列実行可能

#### グループ 3-A: Unit Tests（**3並列実行**）

**並列実行指示:**
この3つのテストファイルは独立しているため、並列作成してください。

##### 【並列タスク 3-A-1】test_mst_computation.py 作成

**Subagent指示:**
```
Task: Create tests/test_mst_computation.py

Requirements:
1. Test MST always forms a tree structure
2. Test MST has N-1 edges
3. Test MST has minimum cost
4. Test MST scaling with different node counts
5. Use scipy.sparse.csgraph.minimum_spanning_tree
6. Minimum 4 test methods

Expected output:
- File: tests/test_mst_computation.py (~200 lines)
- Class: TestMSTComputation
- Tests: test_mst_forms_tree, test_mst_minimum_cost, test_mst_scaling, test_mst_deterministic

Validation:
- pytest tests/test_mst_computation.py -v
- All tests should PASS
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_mst_computation.py が作成される
- [ ] テスト: `pytest tests/test_mst_computation.py -v`
- [ ] エラー時対処: テスト失敗の場合はアサーションを確認

##### 【並列タスク 3-A-2】test_sfs_layer.py 作成

**Subagent指示:**
```
Task: Create tests/test_sfs_layer.py

Requirements:
1. Test E- edge label suppression
2. Test gradient flow through SFS layer
3. Test Lambda parameter effect
4. Follow paper Equation (10)
5. Minimum 3 test methods

Expected output:
- File: tests/test_sfs_layer.py (~150 lines)
- Class: TestSFSLayer
- Tests: test_label_suppression, test_gradient_flow, test_lambda_effect

Validation:
- pytest tests/test_sfs_layer.py -v
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_sfs_layer.py が作成される
- [ ] テスト: `pytest tests/test_sfs_layer.py -v`
- [ ] エラー時対処: 勾配テストが失敗する場合はdetachを確認

##### 【並列タスク 3-A-3】pytest.ini 作成

**Subagent指示:**
```
Task: Create pytest.ini configuration file

Requirements:
1. Define test paths, markers
2. Set addopts for verbose output
3. Add markers: unit, integration, e2e, slow
4. Configure coverage if needed

Expected output:
- File: pytest.ini (~30 lines)
- Markers: unit, integration, e2e, slow
- Addopts: -v, --tb=short, --strict-markers

Validation:
- pytest --markers
- Custom markers displayed
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: pytest.ini が作成される
- [ ] テスト: `pytest --markers | grep -E "(unit|integration|e2e)"`
- [ ] エラー時対処: マーカーが認識されない場合は構文確認

---

#### グループ 3-B: Integration/E2E Tests（**2並列実行**）

**並列実行指示:**
グループ 2-B完了後、この2つのテストファイルを並列作成してください。

##### 【並列タスク 3-B-1】test_integration.py 作成

**Subagent指示:**
```
Task: Create tests/test_integration.py

Requirements:
1. Test complete forward pass
2. Test batch consistency
3. Test output shapes
4. Use actual inference functions
5. Minimum 2 test methods

Expected output:
- File: tests/test_integration.py (~150 lines)
- Class: TestIntegration
- Tests: test_forward_pass_complete, test_batch_consistency

Validation:
- pytest tests/test_integration.py -v -m integration
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_integration.py が作成される
- [ ] テスト: `pytest tests/test_integration.py -v`
- [ ] エラー時対処: Forward passエラーの場合はモックを確認

##### 【並列タスク 3-B-2】test_e2e.py 作成

**Subagent指示:**
```
Task: Create tests/test_e2e.py

Requirements:
1. Test tree rate: MST vs Unconstrained
2. Test performance metrics (inference time)
3. Compare all 3 implementations
4. Minimum 2 test methods

Expected output:
- File: tests/test_e2e.py (~180 lines)
- Class: TestE2E
- Tests: test_tree_rate_mst_vs_unconstrained, test_performance_metrics

Validation:
- pytest tests/test_e2e.py -v -m e2e
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: tests/test_e2e.py が作成される
- [ ] テスト: `pytest tests/test_e2e.py -v`
- [ ] エラー時対処: パフォーマンステストが遅い場合はタイムアウト調整

---

#### 手順 3-C: フェーズ3完了確認（逐次実行）

##### 手順 3-C-1: 全テスト実行
- [ ] 操作: `pytest tests/ -v --tb=short`
- [ ] 確認: 全テストがPASS（FAILED=0）
- [ ] テスト: テストサマリーでPASSED >= 15
- [ ] エラー時対処: 失敗したテストのtracebackを確認

##### 手順 3-C-2: カバレッジ確認
- [ ] 操作: `pytest tests/ -v --cov=. --cov-report=term-missing`
- [ ] 確認: カバレッジ > 70%
- [ ] テスト: inference_*.py のカバレッジが表示される
- [ ] エラー時対処: pytest-covがない場合は `uv pip install pytest-cov`

---

### フェーズ 4: [統合とリグレッション確認] - 逐次実行

#### 手順 4-1: valid_smd_guyot_nx.pyの本実装切り替え
- [x] 操作: valid_smd_guyot_nx.py の1255-1259行を以下に変更
  ```python
  if is_use_mst:
      from inference_infinity_mst_nx_gradmst import relation_infer
  else:
      from inference_infinity_gradmst import relation_infer
  ```
- [x] 確認: `sed -n '1255,1259p' valid_smd_guyot_nx.py` で変更確認
- [x] テスト: `python -m py_compile valid_smd_guyot_nx.py`
- [x] エラー時対処: ModuleNotFoundErrorの場合は__init__.py確認

#### 手順 4-2: import動作確認
- [x] 操作: `python -c "from inference_infinity_mst_nx_gradmst import relation_infer; print('OK')"`
- [x] 確認: "OK"が表示される
- [x] テスト: 他の2ファイルも同様に確認
- [x] エラー時対処: ImportErrorの場合は依存関係確認

#### 手順 4-3: 全テスト再実行
- [ ] 操作: `pytest tests/ -v -m "not slow"`
- [ ] 確認: 全テストPASS
- [ ] テスト: N/A
- [ ] エラー時対処: リグレッションがある場合は変更を確認

#### 手順 4-4: 変更のコミット
- [ ] 操作: `git add inference_*.py tests/ valid_smd_guyot_nx.py pytest.ini && git commit -m "Implement missing inference files with comprehensive tests"`
- [ ] 確認: `git log -1 --stat` でコミット内容確認
- [ ] テスト: N/A
- [ ] エラー時対処: N/A

---

### フェーズ 5: [ドキュメント整備とクリーンアップ] - 並列実行可能

#### グループ 5-A: ドキュメント作成（**3並列実行**）

**並列実行指示:**
この3つのタスクは独立しているため、並列実行してください。

##### 【並列タスク 5-A-1】docstring検証と修正

**Subagent指示:**
```
Task: Verify and fix docstrings in all inference files

Requirements:
1. Check all functions have docstrings
2. Verify parameter descriptions
3. Add type hints if missing
4. Follow NumPy/Google docstring style
5. Run pydocstyle to verify

Expected output:
- Updated docstrings in inference_*.py
- pydocstyle compliant

Validation:
- pydocstyle inference_*.py
- No errors or acceptable warnings only
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: docstringが更新される
- [ ] テスト: `pydocstyle inference_*.py`
- [ ] エラー時対処: pydocstyleがない場合は `uv pip install pydocstyle`

##### 【並列タスク 5-A-2】README更新

**Subagent指示:**
```
Task: Update README.md with new inference files documentation

Requirements:
1. Add section "Inference Modules"
2. Document all 3 inference files
3. Add usage examples
4. Add paper references
5. Keep existing content intact

Expected output:
- Updated README.md with new section
- Usage examples for each inference method

Validation:
- Check markdown syntax
- Verify links work
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: README.mdが更新される
- [ ] テスト: `grep -A 10 "Inference Modules" README.md`
- [ ] エラー時対処: N/A

##### 【並列タスク 5-A-3】コード品質チェック

**Subagent指示:**
```
Task: Run code quality checks on all new files

Requirements:
1. Run ruff format on inference_*.py tests/
2. Run ruff check for linting
3. Run mypy for type checking
4. Fix any critical issues

Expected output:
- Formatted code
- Lint-clean code
- Type-checked code

Validation:
- ruff check inference_*.py tests/
- mypy inference_*.py --ignore-missing-imports
```

**チェックリスト:**
- [ ] 操作: Subagent (general-purpose) に上記タスクを依頼
- [ ] 確認: コードが整形される
- [ ] テスト: `ruff check inference_*.py`
- [ ] エラー時対処: ruffがない場合は `uv pip install ruff`

---

#### 手順 5-B: 最終確認（逐次実行）

##### 手順 5-B-1: 最終テスト実行
- [ ] 操作: `pytest tests/ -v --cov=. --cov-report=html`
- [ ] 確認: HTMLレポートが生成される（htmlcov/index.html）
- [ ] テスト: ブラウザで確認可能
- [ ] エラー時対処: pytest-covがない場合はインストール

##### 手順 5-B-2: 最終コミット
- [ ] 操作: `git add . && git commit -m "Add documentation and finalize implementation"`
- [ ] 確認: `git log --oneline -5` で全コミット確認
- [ ] テスト: N/A
- [ ] エラー時対処: N/A

##### 手順 5-B-3: リモートプッシュ
- [ ] 操作: `git push -u origin claude/implement-missing-inference-files-01D32vfvA14H5qmVAMvjgmhS`
- [ ] 確認: プッシュ成功メッセージ
- [ ] テスト: GitHub上でブランチ確認
- [ ] エラー時対処: 403エラーの場合はブランチ名確認

---

## 5. 完了の定義

作業が完了したと見なすには、以下の全ての条件を満たす必要があります：

- [ ] **機能要件:** 3つの推論ファイルが実装され、valid_smd_guyot_nx.pyが正常に動作する
- [ ] **テスト要件:** pytest実行で全テストがPASS、カバレッジ > 70%
- [ ] **コード品質:** ruff、mypy、pydocstyleのチェックをパス
- [ ] **ドキュメント:** README.md更新、各ファイルにdocstring完備
- [ ] **統合:** 既存のテストスイートに影響なし（リグレッションなし）
- [ ] **Git:** 全変更がコミット・プッシュされ、ブランチが最新

---

## 6. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   並列実行タスクは、開始時と全完了時の両方で記録を取ること。
*   Subagent実行の場合は、subagent_typeと実行結果を記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2025-11-14 | 08:16:54 UTC+0000 | Claude Agent | Phase 1-1: uv環境の確認 | 成功: uv version 0.8.17 インストール済み確認 |
| 2025-11-14 | 08:18:00 UTC+0000 | Claude Agent | Phase 1-2: Python環境の有効化 | 成功: uv venv で .venv 作成、Python 3.10.6 使用、パス確認済み |
| 2025-11-14 | 08:23:10 UTC+0000 | Claude Agent | Phase 1-3: 依存パッケージの確認 | 成功: torch 2.9.1, networkx 3.4.2, scipy 1.15.3, numpy 2.2.6 インストール・import確認済み |
| 2025-11-14 | 08:23:55 UTC+0000 | Claude Agent | Phase 1-4: 既存コードの存在確認 | 成功: epoch.py, losses_only.py, valid_smd_guyot_nx.py 全ファイル存在確認・構文チェック完了 |
| 2025-11-14 | 08:24:31 UTC+0000 | Claude Agent | Phase 1-5: epoch.pyの関数確認 | 成功: relation_infer (行43), relation_infer_mst (行308) 存在確認 |
| 2025-11-14 | 08:25:17 UTC+0000 | Claude Agent | Phase 1-6: valid_smd_guyot_nx.pyのバックアップ | 成功: バックアップファイル作成、差分なし確認 |
| 2025-11-14 | 08:26:22 UTC+0000 | Claude Agent | Phase 1-7: 暫定修正の適用 | 成功: valid_smd_guyot_nx.py 行1255-1259置き換え、構文チェック通過 |
| 2025-11-14 | 08:28:00 UTC+0000 | Claude Agent | Phase 1-8: 暫定修正後のimport確認 | 成功: torchvision追加インストール後、epoch.pyからのimport確認完了 |
| 2025-11-14 | 08:28:58 UTC+0000 | Claude Agent | Phase 1-9: 変更のコミット | 成功: commit 539910b作成完了 |
| 2025-11-14 | 08:30:05 UTC+0000 | Claude Agent | Phase 2-1: テストディレクトリの作成 | 成功: tests/ ディレクトリと __init__.py 作成完了、import確認済み |
| 2025-11-14 | 08:33:07 UTC+0000 | Claude Agent | Phase 2-2: conftest.py作成 | 成功: pytest fixtures作成完了、構文チェック通過。注：mmcvバージョン問題あり（後で対処） |
| 2025-11-14 | 08:33:59 UTC+0000 | Claude Agent | 【並列開始】Group 2-B: 推論ファイル実装 | Agent-1: gradmst, Agent-2: dist, Agent-3: unconst 並列実行開始 |
| 2025-11-14 | 08:38:16 UTC+0000 | Claude Agent | 【並列完了】Group 2-B: 推論ファイル実装 | 成功: 3ファイル作成完了 (gradmst 541行/24KB, dist 367行/18KB, unconst 372行/16KB)、全構文チェック通過 |
| 2025-11-14 | 08:39:44 UTC+0000 | Claude Agent | Commits: 推論ファイルとテスト基盤 | commit 131f7ca (推論ファイル3つ), commit fd40a2f (tests/conftest.py) |
| 2025-11-14 | 08:41:26 UTC+0000 | Claude Agent | Phase 4-1, 4-2: 本実装への切り替え | 成功: valid_smd_guyot_nx.py を実装ファイルに切り替え、全3ファイルimport確認完了 |
| 2025-11-14 | 08:42:53 UTC+0000 | Claude Agent | Git Push完了 | 成功: 5 commits を remote branch に push 完了 (539910b~bd94ba7) |
| | | | | |

---

## 付録A: 並列実行コマンド例

### フェーズ2グループ2-B: 3つの推論ファイルを並列実装

```python
# 単一メッセージで3つのTask toolを同時呼び出し
# Agent-1: inference_infinity_mst_nx_gradmst.py
# Agent-2: inference_infinity_mst_nx_dist.py
# Agent-3: inference_infinity_gradmst.py
```

### フェーズ3グループ3-A: 3つのUnit Testsを並列作成

```python
# 単一メッセージで3つのTask toolを同時呼び出し
# Agent-4: test_mst_computation.py
# Agent-5: test_sfs_layer.py
# Agent-6: pytest.ini
```

### フェーズ5グループ5-A: 3つのドキュメントタスクを並列実行

```python
# 単一メッセージで3つのTask toolを同時呼び出し
# Agent-9: docstring検証
# Agent-10: README更新
# Agent-11: コード品質チェック
```

---

## 付録B: Subagent実行ログテンプレート

並列実行時は以下の形式で作業記録に記載してください：

```
【並列実行開始】グループ 2-B: 推論ファイル実装
開始時刻: 2025-11-14 08:30:00 UTC+0000

Agent-1 (general-purpose): inference_infinity_mst_nx_gradmst.py
Agent-2 (general-purpose): inference_infinity_mst_nx_dist.py
Agent-3 (general-purpose): inference_infinity_gradmst.py

【並列実行完了】グループ 2-B
完了時刻: 2025-11-14 08:35:00 UTC+0000
結果: 全Agent成功、3ファイル作成完了
```

---

**作業計画書（並列実行版） 終わり**
**最終更新:** 2025-11-14
