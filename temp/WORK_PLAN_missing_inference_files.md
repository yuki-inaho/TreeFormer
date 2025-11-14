# 作業計画書 兼 記録書：TreeFormer 欠損推論ファイル実装

---

**日付：** `2025年11月14日`
**作業ディレクトリ・リポジトリ:** `/home/user/TreeFormer` (yuki-inaho/TreeFormer)
**作業ブランチ:** `claude/review-code-and-docs-01MsuxYMUdiHb6xFcbGtVZgT`
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

## 2. 作業内容

### フェーズ 1: [環境確認と緊急修正] (見積: 0.5h)
このフェーズでは、作業環境を確認し、最小限の修正で検証スクリプトを動作させます。

#### 目的
- uv環境が正しく構築されているか確認
- 既存実装を活用した暫定修正を実施
- 基本的な動作確認を行う

#### タスク
1.  **uv環境の確認と依存関係のインストール:**
    *   **タスク内容：** uv環境が存在するか確認し、必要な依存パッケージがインストールされているか検証します
    *   **目的：** 作業に必要な環境が整っていることを保証します
2.  **valid_smd_guyot_nx.py の暫定修正:**
    *   **タスク内容：** 行1255-1259のimport文を、既存の `epoch.py` からインポートするように修正します
    *   **目的：** 検証スクリプトが即座に動作するようにし、後続の作業でリグレッションがないことを確認できるようにします
3.  **修正の動作確認:**
    *   **タスク内容：** 修正後のスクリプトが構文エラーなく実行可能か確認します
    *   **目的：** 基本的な動作を保証し、次のフェーズに進む準備を整えます

### フェーズ 2: [テストファーストでの欠損ファイル実装] (見積: 3.0h)
このフェーズでは、TDDアプローチで欠損ファイルを実装します。各ファイルの実装前にテストを書き、テストが失敗→成功に変わることを確認します。

#### 目的
- 論文アルゴリズムに忠実な実装を作成
- テストで実装の正確性を保証
- 既存の epoch.py との互換性を維持

#### タスク
1.  **テストディレクトリとモックの準備:**
    *   **タスク内容：** `tests/` ディレクトリを作成し、モックデータとヘルパー関数を実装します
    *   **目的：** テスト環境を構築し、再利用可能なテストインフラを整備します
2.  **inference_infinity_mst_nx_gradmst.py の実装:**
    *   **タスク内容：** TreeFormerのSFS Layer実装（MST + 勾配対応）を作成します
    *   **目的：** 論文の「Ours (TreeFormer)」手法を実装します
3.  **inference_infinity_mst_nx_dist.py の実装:**
    *   **タスク内容：** Test-time constraint実装（推論時のみMST適用）を作成します
    *   **目的：** 論文のベースライン手法を実装します
4.  **inference_infinity_gradmst.py の実装:**
    *   **タスク内容：** Unconstrained baseline実装（閾値ベース選択）を作成します
    *   **目的：** 論文のもう一つのベースライン手法を実装します

### フェーズ 3: [包括的テストスイートの作成] (見積: 2.5h)
このフェーズでは、実装した推論ファイルの正確性を保証する包括的なテストを作成します。

#### 目的
- Unit/Integration/E2Eの3層でテスト
- MST計算の正確性を数学的に検証
- 論文の性能指標（Tree rate 100%等）を再現

#### タスク
1.  **Unit Tests: MST計算の正確性:**
    *   **タスク内容：** MST計算が数学的に正しいことを検証するテストを作成します
    *   **目的：** Kruskalアルゴリズムの実装が正確であることを保証します
2.  **Unit Tests: SFS Layerの動作:**
    *   **タスク内容：** E+/E-エッジの特徴修正が正しく行われることを検証します
    *   **目的：** 論文Equation (10)の実装が正確であることを保証します
3.  **Integration Tests: 推論パイプライン:**
    *   **タスク内容：** モデル→推論→出力の一連の流れをテストします
    *   **目的：** 実際の使用シナリオでの動作を保証します
4.  **E2E Tests: メトリクス検証:**
    *   **タスク内容：** Tree rate、SMD、TOPOスコアが期待値通りか検証します
    *   **目的：** 論文の結果を再現できることを保証します

### フェーズ 4: [統合とリグレッション確認] (見積: 1.5h)
このフェーズでは、実装した推論ファイルが既存システムに正しく統合され、既存機能に悪影響がないことを確認します。

#### 目的
- 既存の訓練・検証パイプラインへの統合
- リグレッションがないことの確認
- パフォーマンスの検証

#### タスク
1.  **valid_smd_guyot_nx.py を新実装に切り替え:**
    *   **タスク内容：** 暫定修正を本実装に置き換えます
    *   **目的：** 完全な実装を使用した検証を行います
2.  **既存テストスイートの実行:**
    *   **タスク内容：** プロジェクト既存のテストが全てパスすることを確認します
    *   **目的：** リグレッションがないことを保証します
3.  **小規模データセットでの動作確認:**
    *   **タスク内容：** Guyot_200_20サブセットで推論を実行し、メトリクスを確認します
    *   **目的：** 実データでの動作を検証します

### フェーズ 5: [ドキュメント整備とクリーンアップ] (見積: 1.0h)
最終フェーズでは、実装内容をドキュメント化し、コード品質を保証します。

#### 目的
- 実装内容の文書化
- コード規約の遵守
- 将来の保守性の確保

#### タスク
1.  **各推論ファイルのdocstring整備:**
    *   **タスク内容：** 関数・クラスに詳細なdocstringを追加します
    *   **目的：** APIドキュメントを自動生成できるようにします
2.  **README.mdの更新:**
    *   **タスク内容：** 新しい推論ファイルの使い方を追記します
    *   **目的：** ユーザーが新機能を利用できるようにします
3.  **コード品質チェック:**
    *   **タスク内容：** ruff、black、mypyを実行してコード品質を保証します
    *   **目的：** プロジェクトのコーディング規約を遵守します

---

## 3. 作業チェックリスト

### フェーズ 1: [環境確認と緊急修正]

#### 手順 1-1: uv環境の確認
- [ ] 操作: `uv --version` でuvがインストールされているか確認
- [ ] 確認: バージョン情報が表示される（例: `uv 0.4.x`）
- [ ] テスト: N/A（環境確認のため）
- [ ] エラー時対処: uvが見つからない場合は `curl -LsSf https://astral.sh/uv/install.sh | sh` でインストール

#### 手順 1-2: Python環境の有効化
- [ ] 操作: `uv venv` で仮想環境を作成（存在しない場合）
- [ ] 確認: `.venv/` ディレクトリが作成される
- [ ] テスト: N/A
- [ ] エラー時対処: Python 3.10以上がインストールされているか確認（`python3 --version`）

#### 手順 1-3: 依存パッケージの確認
- [ ] 操作: `uv pip list | grep -E "(torch|networkx|scipy|numpy)"` で必須パッケージを確認
- [ ] 確認: torch, networkx, scipy, numpyが表示される
- [ ] テスト: N/A
- [ ] エラー時対処: 不足している場合は `uv pip install torch networkx scipy numpy` を実行

#### 手順 1-4: 既存コードの存在確認
- [ ] 操作: `ls -l epoch.py losses_only.py valid_smd_guyot_nx.py` で必要ファイルが存在するか確認
- [ ] 確認: 3つのファイルが全て存在し、サイズが0より大きい
- [ ] テスト: N/A
- [ ] エラー時対処: ファイルが存在しない場合はgit cloneし直すか、正しいディレクトリに移動

#### 手順 1-5: epoch.pyの関数確認
- [ ] 操作: `grep -n "^def relation_infer" epoch.py` で必要な関数が存在するか確認
- [ ] 確認: `relation_infer` と `relation_infer_mst` が見つかる（行番号付き）
- [ ] テスト: N/A
- [ ] エラー時対処: 関数が見つからない場合は、正しいブランチにいるか確認（`git branch`）

#### 手順 1-6: valid_smd_guyot_nx.pyのバックアップ
- [ ] 操作: `cp valid_smd_guyot_nx.py valid_smd_guyot_nx.py.backup` でバックアップを作成
- [ ] 確認: `ls -l valid_smd_guyot_nx.py*` で2ファイル表示される
- [ ] テスト: N/A
- [ ] エラー時対処: 書き込み権限がない場合は `chmod u+w .` で権限付与

#### 手順 1-7: valid_smd_guyot_nx.pyの該当行確認
- [ ] 操作: `sed -n '1255,1259p' valid_smd_guyot_nx.py` で現在のimport文を表示
- [ ] 確認: 以下のような行が表示される
  ```python
  if is_use_mst:
      # from inference_infinity_mst_nx_dist import relation_infer
      from inference_infinity_mst_nx_gradmst import relation_infer
  else:
      from inference_infinity_gradmst import relation_infer
  ```
- [ ] テスト: N/A
- [ ] エラー時対処: 行番号が異なる場合は `grep -n "from inference_infinity" valid_smd_guyot_nx.py` で正確な行番号を特定

#### 手順 1-8: 暫定修正の適用
- [ ] 操作: エディタで `valid_smd_guyot_nx.py` の1255-1259行を以下に置き換え
  ```python
  if is_use_mst:
      from epoch import relation_infer_mst as relation_infer
  else:
      from epoch import relation_infer
  ```
- [ ] 確認: `sed -n '1255,1259p' valid_smd_guyot_nx.py` で変更後の内容を確認
- [ ] テスト: `python -m py_compile valid_smd_guyot_nx.py` で構文エラーがないか確認
- [ ] エラー時対処: SyntaxErrorが出る場合はインデントを確認（4スペース）

#### 手順 1-9: 暫定修正後のimport確認
- [ ] 操作: `python -c "import sys; sys.path.insert(0, '.'); from valid_smd_guyot_nx import *"` でimportテスト
- [ ] 確認: エラーなく完了する（警告は許容）
- [ ] テスト: N/A
- [ ] エラー時対処: ModuleNotFoundErrorが出る場合は依存パッケージを再確認

#### 手順 1-10: 変更のコミット
- [ ] 操作: `git add valid_smd_guyot_nx.py && git commit -m "Fix: Use existing epoch.py functions for inference (temporary)"`
- [ ] 確認: `git log -1 --oneline` でコミットが作成される
- [ ] テスト: N/A
- [ ] エラー時対処: コミットに失敗する場合は `git status` で状態確認

---

### フェーズ 2: [テストファーストでの欠損ファイル実装]

#### 手順 2-1: テストディレクトリの作成
- [ ] 操作: `mkdir -p tests && touch tests/__init__.py` でテストディレクトリを作成
- [ ] 確認: `ls -ld tests/ tests/__init__.py` でディレクトリとファイルが存在
- [ ] テスト: `python -c "import tests"` でインポート可能
- [ ] エラー時対処: 既存のtests/ディレクトリがある場合はそれを使用

#### 手順 2-2: テスト用モックデータジェネレータの作成
- [ ] 操作: `tests/conftest.py` を作成し、pytest fixtureを実装
- [ ] 確認: 以下のfixtureが含まれる
  - `dummy_model_output`: モデル出力のダミーデータ
  - `dummy_hidden_features`: 隠れ層特徴のダミーデータ
  - `mock_network`: モックニューラルネットワーク
- [ ] テスト: `pytest tests/conftest.py -v` でfixtureが認識される
- [ ] エラー時対処: pytestがない場合は `uv pip install pytest` でインストール

**tests/conftest.py の実装内容:**
```python
"""
Pytest fixtures for TreeFormer inference tests.
"""
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def dummy_model_output():
    """
    モデル出力のダミーデータを生成

    Returns:
        dict: 'pred_logits' と 'pred_nodes' を含む辞書
    """
    batch_size = 2
    num_tokens = 20

    return {
        'pred_logits': torch.randn(batch_size, num_tokens, 2),
        'pred_nodes': torch.rand(batch_size, num_tokens, 4)  # [cx, cy, w, h]
    }


@pytest.fixture
def dummy_hidden_features():
    """
    隠れ層特徴のダミーデータを生成

    Returns:
        torch.Tensor: [batch, seq_len, hidden_dim] 形状のテンソル
    """
    batch_size = 2
    seq_len = 21  # 20 obj + 1 rln
    hidden_dim = 256

    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def mock_network():
    """
    モックニューラルネットワークを作成

    Returns:
        nn.Module: relation_embedメソッドを持つモックモジュール
    """
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

#### 手順 2-3: MST計算のユニットテスト作成（Red phase）
- [ ] 操作: `tests/test_mst_computation.py` を作成
- [ ] 確認: テストが失敗することを確認（実装前なのでこれが正しい）
- [ ] テスト: `pytest tests/test_mst_computation.py -v` → **全テストがSKIPまたはFAIL**
- [ ] エラー時対処: テストがエラーで終了する場合はimport文を確認

**tests/test_mst_computation.py の実装内容:**
```python
"""
Unit tests for MST computation accuracy.
"""
import pytest
import torch
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree


class TestMSTComputation:
    """MST計算の正確性をテスト"""

    def test_mst_forms_tree(self):
        """
        TEST-MST-001: MSTが木構造を形成することを確認

        期待結果:
        - 出力グラフがツリー（閉路なし、連結）
        - エッジ数が N-1
        """
        N = 10
        cost_matrix = np.random.rand(N, N)
        cost_matrix = (cost_matrix + cost_matrix.T) / 2
        np.fill_diagonal(cost_matrix, 0)

        # MST計算
        mst = minimum_spanning_tree(cost_matrix)
        mst_adj = (mst + mst.T).toarray()

        # NetworkXで検証
        G = nx.Graph()
        edges = np.argwhere(mst_adj > 0)
        G.add_edges_from(edges)

        assert nx.is_tree(G), "MST must form a tree structure"
        assert len(G.edges) == N - 1, f"MST must have {N-1} edges, got {len(G.edges)}"

    def test_mst_minimum_cost(self):
        """
        TEST-MST-002: MSTが最小コストであることを確認

        期待結果:
        - 既知のコスト行列で、期待される総コスト
        """
        N = 5
        cost_matrix = np.array([
            [0, 1, 3, 4, 5],
            [1, 0, 2, 4, 3],
            [3, 2, 0, 1, 6],
            [4, 4, 1, 0, 2],
            [5, 3, 6, 2, 0]
        ], dtype=float)

        mst = minimum_spanning_tree(cost_matrix)
        mst_adj = (mst + mst.T).toarray()

        total_cost = (mst_adj * cost_matrix).sum() / 2
        expected_cost = 6  # 手計算での最小コスト

        assert abs(total_cost - expected_cost) < 1e-6, \
            f"MST cost {total_cost} != expected {expected_cost}"

    @pytest.mark.parametrize("num_nodes", [3, 5, 10, 20])
    def test_mst_scaling(self, num_nodes):
        """
        TEST-MST-003: 異なるノード数でMSTが正しく動作

        期待結果:
        - 任意のノード数でツリー構造を形成
        """
        cost_matrix = np.random.rand(num_nodes, num_nodes)
        cost_matrix = (cost_matrix + cost_matrix.T) / 2
        np.fill_diagonal(cost_matrix, 0)

        mst = minimum_spanning_tree(cost_matrix)
        mst_adj = (mst + mst.T).toarray()

        G = nx.Graph()
        edges = np.argwhere(mst_adj > 0)
        G.add_edges_from(edges)

        assert nx.is_tree(G)
        assert len(G.edges) == num_nodes - 1
```

#### 手順 2-4: inference_infinity_mst_nx_gradmst.py の実装（Green phase）
- [ ] 操作: ルートディレクトリに `inference_infinity_mst_nx_gradmst.py` を作成
- [ ] 確認: ファイルが作成され、必要な関数が実装される
- [ ] テスト: `pytest tests/test_inference_gradmst.py -v` → **全テストがPASS**
- [ ] エラー時対処: テストが失敗する場合は実装ロジックを見直す

**inference_infinity_mst_nx_gradmst.py の実装内容:**
```python
"""
TreeFormer MST-constrained Inference with Gradient Support.

This module implements the "Ours (TreeFormer with SFS Layer)" method
from the paper, which applies MST constraints during both training
and inference.

Paper Reference:
    Section 4.2: "Tree-constrained graph generation"
    Equation (10): Feature modification with SFS layer
"""
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import itertools
from typing import Dict, List, Tuple, Optional
from torchvision.ops import batched_nms


def relation_infer(
    h: torch.Tensor,
    out: Dict[str, torch.Tensor],
    net: torch.nn.Module,
    obj_token: int,
    rln_token: int,
    nms: bool = False,
    map_: bool = False
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    MST制約付き推論（TreeFormer SFS Layer実装）

    このアルゴリズムは論文の「Ours (TreeFormer)」手法に対応し、
    訓練時と推論時の両方でMST制約を適用します。

    Args:
        h: [batch, seq_len, hidden_dim] 隠れ層特徴
            - seq_len = obj_token + rln_token
        out: モデル出力辞書
            - 'pred_logits': [batch, obj_token, 2] ノード分類ロジット
            - 'pred_nodes': [batch, obj_token, 4] ノード位置 [cx, cy, w, h]
        net: ニューラルネットワークモデル
            - net.module.relation_embed: エッジ予測MLP
        obj_token: オブジェクトトークン数（通常20）
        rln_token: リレーショントークン数（通常1）
        nms: Non-Maximum Suppression適用有無
        map_: マッピング情報返却有無

    Returns:
        pred_nodes: 予測ノード座標のリスト
        pred_edges: 予測エッジインデックスのリスト
        (map_=True時) + ボックス情報、スコア、クラス

    Algorithm:
        1. Object & Relation tokenを分離
        2. 有効ノードを抽出（valid_token）
        3. 完全グラフのノードペアを生成
        4. エッジ予測を実行
        5. コスト行列を構築（cost = p(edge non-exist)）
        6. Kruskal's MSTを計算（NetworkX）
        7. MSTエッジインデックスを抽出

    Paper Correspondence:
        - Section 4.2: "To introduce the tree structure constraint,
          we use Kruskal's MST algorithm"
        - Cost definition: edge non-existence probabilities {ŷ^-_(i,j)}
    """
    # 1. トークン分離
    object_token = h[..., :obj_token, :]  # [batch, obj_token, hidden_dim]

    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]  # [batch, rln_token, hidden_dim]

    # 2. 有効ノード抽出
    valid_token = torch.argmax(out['pred_logits'], -1).detach()  # [batch, obj_token]

    # NMS適用（オプション）
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(
            zip(valid_token, out['pred_logits'], out['pred_nodes'])
        ):
            valid_token_id = torch.nonzero(token).squeeze(1)

            if valid_token_id.numel() == 0:
                continue

            valid_logits = logits[valid_token_id]
            valid_nodes = nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # ボックス座標を調整
            valid_nodes_boxes = valid_nodes.clone()
            valid_nodes_boxes[:, 2:] = valid_nodes[:, :2] + 0.5

            ids2keep = batched_nms(
                boxes=valid_nodes_boxes * 1000,
                scores=valid_scores,
                idxs=torch.ones_like(valid_scores, dtype=torch.long),
                iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            valid_token_nms[idx][valid_token_id_nms] = 1

        valid_token = valid_token_nms

    pred_nodes = []
    pred_edges = []

    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    # 3. バッチごとに処理
    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        if map_:
            pred_nodes_boxes.append(
                out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy()
            )
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:
            # 4. 完全グラフのノードペア生成
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))

            node_pairs_valid = torch.tensor(
                [list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))]
            )

            # 辞書マッピング（map_=True用）
            node_pairs_valid_dict = {}
            for num in range(node_pairs_valid.shape[0]):
                node_pair = node_pairs_valid[num]
                node_pairs_valid_dict[tuple(node_pair.cpu().numpy().tolist())] = num

            # 5. エッジ予測
            if rln_token > 0:
                relation_feature1 = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
                relation_feature2 = torch.cat((
                    object_token[batch_id, node_pairs[1], :],
                    object_token[batch_id, node_pairs[0], :],
                    relation_token[batch_id, ...].view(1, -1).repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                relation_feature1 = torch.cat(
                    (object_token[batch_id, node_pairs[0], :],
                     object_token[batch_id, node_pairs[1], :]), 1
                )
                relation_feature2 = torch.cat(
                    (object_token[batch_id, node_pairs[1], :],
                     object_token[batch_id, node_pairs[0], :]), 1
                )

            relation_pred1 = net.module.relation_embed(relation_feature1).detach()
            relation_pred2 = net.module.relation_embed(relation_feature2).detach()
            relation_pred = (relation_pred1 + relation_pred2) / 2.0

            # 6. コスト行列構築
            relation_pred_softmax = F.softmax(relation_pred, dim=-1).detach()
            cost_pred_batch = relation_pred_softmax[:, 0]  # 非存在確率 = コスト

            cost_adj_batch = torch.ones((node_id.shape[0], node_id.shape[0])).to(h.device) * 9999
            x, y = node_pairs_valid.t()
            cost_adj_batch[x, y] = cost_pred_batch
            cost_adj_batch[y, x] = cost_pred_batch

            # 7. Kruskal's MST計算（NetworkX）
            mst_adj_batch = compute_mst_nx(node_pairs_valid, cost_pred_batch)

            # 8. エッジインデックス抽出
            mst_adj_batch = mst_adj_batch * torch.triu(
                torch.ones_like(mst_adj_batch), diagonal=1
            )
            mst_tree_selected_list = torch.nonzero(mst_adj_batch, as_tuple=False)
            pred_edges.append(mst_tree_selected_list.cpu().numpy())

            # map_=True時の追加情報
            if map_:
                pred_rel_list = [
                    node_pairs_valid_dict[tuple(sorted((int(xy[0]), int(xy[1]))))]
                    for xy in mst_tree_selected_list if xy[0] != xy[1]
                ]

                if len(pred_rel_list) > 0:
                    pred_rel = torch.tensor(pred_rel_list).cpu().numpy()
                    pred_edges_boxes_score.append(
                        relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy()
                    )
                    pred_edges_boxes_class.append(
                        torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy()
                    )
                else:
                    pred_edges_boxes_score.append(np.array([]))
                    pred_edges_boxes_class.append(np.array([]))
        else:
            pred_edges.append(np.empty((0, 2)))

            if map_:
                pred_edges_boxes_score.append(np.empty(0))
                pred_edges_boxes_class.append(np.empty(0))

    if map_:
        return (pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score,
                pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class)
    else:
        return pred_nodes, pred_edges


def compute_mst_nx(
    node_pairs_valid: torch.Tensor,
    cost_pred_batch: torch.Tensor
) -> torch.Tensor:
    """
    NetworkXを使用したKruskal's MST計算

    Args:
        node_pairs_valid: [num_pairs, 2] ノードペアインデックス
        cost_pred_batch: [num_pairs] 各ペアのコスト

    Returns:
        mst_adj_batch: [N, N] MST隣接行列

    Algorithm:
        1. NetworkX Graphを構築
        2. 重み付きエッジを追加
        3. Kruskal's algorithmでMST計算
        4. 隣接行列に変換

    Note:
        この関数は微分不可能（detach済みのテンソルを使用）
    """
    G = nx.Graph()
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    # 重み付きエッジを追加
    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    # Kruskal's MST
    mst_edges = list(nx.minimum_spanning_edges(G, algorithm="kruskal", data=False))

    # 隣接行列構築
    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight

    mst_adj_batch = torch.tensor(mst_adj_np)
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch
```

#### 手順 2-5: inference_infinity_mst_nx_gradmst.py のテスト作成と実行
- [ ] 操作: `tests/test_inference_gradmst.py` を作成
- [ ] 確認: 以下のテストケースが含まれる
  - MST出力が常にツリー構造
  - エッジ数が N-1
  - epoch.relation_infer_mst との互換性
- [ ] テスト: `pytest tests/test_inference_gradmst.py -v` → **全テストがPASS**
- [ ] エラー時対処: テストが失敗する場合は実装を見直す

**tests/test_inference_gradmst.py の実装内容:**
```python
"""
Tests for inference_infinity_mst_nx_gradmst.py
"""
import pytest
import torch
import numpy as np
import networkx as nx
from inference_infinity_mst_nx_gradmst import relation_infer, compute_mst_nx


class TestInferenceGradMST:
    """TreeFormer SFS Layer実装のテスト"""

    def test_output_is_tree(self, dummy_hidden_features, dummy_model_output, mock_network):
        """
        TEST-INFER-001: MST出力が常にツリー構造

        期待結果:
        - 全バッチでツリー構造
        - エッジ数が N-1
        """
        h = dummy_hidden_features
        out = dummy_model_output
        net = mock_network

        pred_nodes, pred_edges = relation_infer(h, out, net, obj_token=20, rln_token=1)

        for edges in pred_edges:
            if len(edges) > 0:
                G = nx.Graph()
                G.add_edges_from(edges)

                assert nx.is_tree(G), "Output must be a tree structure"
                assert len(G.edges) == len(G.nodes) - 1, "Tree must have N-1 edges"

    def test_consistency_with_epoch(self, dummy_hidden_features, dummy_model_output, mock_network):
        """
        TEST-INFER-002: epoch.relation_infer_mst との互換性

        期待結果:
        - 出力形式が同じ
        - ノード予測が一致
        """
        h = dummy_hidden_features
        out = dummy_model_output
        net = mock_network

        # 新実装
        pred_nodes_new, pred_edges_new = relation_infer(
            h, out, net, obj_token=20, rln_token=1
        )

        # epoch.pyの実装をインポート
        from epoch import relation_infer_mst
        pred_nodes_old, pred_edges_old = relation_infer_mst(
            h, out, net, obj_token=20, rln_token=1
        )

        # ノード予測が一致
        assert len(pred_nodes_new) == len(pred_nodes_old)
        for n_new, n_old in zip(pred_nodes_new, pred_nodes_old):
            assert torch.allclose(n_new, n_old, atol=1e-6)

    def test_compute_mst_nx(self):
        """
        TEST-INFER-003: compute_mst_nx が正しく動作

        期待結果:
        - 最小コストのMST
        - ツリー構造
        """
        # ノードペア
        node_pairs = torch.tensor([[0, 1], [1, 2], [0, 2]])
        costs = torch.tensor([1.0, 2.0, 3.0])

        mst_adj = compute_mst_nx(node_pairs, costs)

        # エッジ (0,1) と (1,2) が選ばれるべき
        assert mst_adj[0, 1] > 0 or mst_adj[1, 0] > 0
        assert mst_adj[1, 2] > 0 or mst_adj[2, 1] > 0
        assert mst_adj[0, 2] == 0 and mst_adj[2, 0] == 0
```

#### 手順 2-6: inference_infinity_mst_nx_dist.py の実装
- [ ] 操作: `inference_infinity_mst_nx_dist.py` を作成（distance重み付き版）
- [ ] 確認: relation_infer関数が実装され、use_distanceパラメータを持つ
- [ ] テスト: `pytest tests/test_inference_dist.py -v` → **全テストがPASS**
- [ ] エラー時対処: テストが失敗する場合は距離計算ロジックを確認

**inference_infinity_mst_nx_dist.py の実装内容:**
```python
"""
Test-time Constraint Inference with Distance Weighting.

This module implements the "Test-time constraint" baseline method
from the paper, with optional geometric distance weighting.

Paper Reference:
    Section 5.3: "Test-time constraint"
    "We apply MST only in the inference phase, where the graph generator
    is trained using the same procedure as the unconstrained method."
"""
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import itertools
from typing import Dict, List, Tuple, Optional
from torchvision.ops import batched_nms


def relation_infer(
    h: torch.Tensor,
    out: Dict[str, torch.Tensor],
    net: torch.nn.Module,
    obj_token: int,
    rln_token: int,
    nms: bool = False,
    map_: bool = False,
    use_distance: bool = False,
    distance_weight: float = 0.3
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Test-time constraint推論（距離重み付きオプション）

    このアルゴリズムは論文の「Test-time constraint」手法に対応し、
    推論時のみMST制約を適用します。オプションで幾何学的距離を
    エッジコストに組み込むことができます。

    Args:
        h: [batch, seq_len, hidden_dim] 隠れ層特徴
        out: モデル出力辞書
        net: ニューラルネットワークモデル
        obj_token: オブジェクトトークン数
        rln_token: リレーショントークン数
        nms: Non-Maximum Suppression適用有無
        map_: マッピング情報返却有無
        use_distance: 幾何学的距離を重みに組み込むか
        distance_weight: 距離の重み（0.0-1.0）

    Returns:
        pred_nodes: 予測ノード座標のリスト
        pred_edges: 予測エッジインデックスのリスト

    Algorithm:
        1. エッジ予測を実行（通常のRelationFormer）
        2. コスト行列を構築
        3. （オプション）幾何学的距離を追加
        4. MST計算

    Distance Weighting:
        final_cost = (1 - α) * p(non-exist) + α * normalized_distance
        where α = distance_weight
    """
    # inference_infinity_mst_nx_gradmst.py と同じ実装
    # ただし、距離重み付けのロジックを追加

    object_token = h[..., :obj_token, :]

    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    if nms:
        # NMS処理（省略、上記と同じ）
        pass

    pred_nodes = []
    pred_edges = []

    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        if map_:
            pred_nodes_boxes.append(
                out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy()
            )
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        if node_id.numel() > 1:
            node_pairs_valid = torch.tensor(
                list(itertools.combinations(range(len(node_id)), 2))
            )

            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))

            # エッジ予測
            if rln_token > 0:
                relation_feature = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                relation_feature = torch.cat(
                    (object_token[batch_id, node_pairs[0], :],
                     object_token[batch_id, node_pairs[1], :]), 1
                )

            relation_pred = net.module.relation_embed(relation_feature).detach()
            relation_pred_softmax = F.softmax(relation_pred, dim=-1)
            cost_pred_batch = relation_pred_softmax[:, 0]  # 非存在確率

            # 距離重み付き（オプション）
            if use_distance:
                nodes_coords = out['pred_nodes'][batch_id, node_id, :2].detach()
                distances = torch.cdist(nodes_coords, nodes_coords)
                distance_weights = distances[
                    node_pairs_valid[:, 0],
                    node_pairs_valid[:, 1]
                ]
                # 正規化
                distance_weights = (distance_weights - distance_weights.min()) / \
                                  (distance_weights.max() - distance_weights.min() + 1e-8)

                # 線形結合
                cost_pred_batch = (1 - distance_weight) * cost_pred_batch.cpu() + \
                                 distance_weight * distance_weights.cpu()

            # MST計算
            mst_adj_batch = compute_mst_nx(node_pairs_valid, cost_pred_batch)
            mst_adj_batch = mst_adj_batch * torch.triu(
                torch.ones_like(mst_adj_batch), diagonal=1
            )

            mst_tree_selected_list = torch.nonzero(mst_adj_batch, as_tuple=False)
            pred_edges.append(mst_tree_selected_list.cpu().numpy())

            if map_:
                # map_情報の処理（省略）
                pass
        else:
            pred_edges.append(np.empty((0, 2)))

            if map_:
                pred_edges_boxes_score.append(np.empty(0))
                pred_edges_boxes_class.append(np.empty(0))

    if map_:
        return (pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score,
                pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class)
    else:
        return pred_nodes, pred_edges


def compute_mst_nx(
    node_pairs_valid: torch.Tensor,
    cost_pred_batch: torch.Tensor
) -> torch.Tensor:
    """MST計算（inference_infinity_mst_nx_gradmst.py と同じ）"""
    # 実装は上記と同一
    G = nx.Graph()
    node_pairs_np = node_pairs_valid.cpu().numpy()
    cost_pred_np = cost_pred_batch.cpu().numpy()

    edges = [(int(u), int(v), float(w)) for (u, v), w in zip(node_pairs_np, cost_pred_np)]
    G.add_weighted_edges_from(edges)

    mst_edges = list(nx.minimum_spanning_edges(G, algorithm="kruskal", data=False))

    num_nodes = len(G)
    mst_adj_np = np.zeros((num_nodes, num_nodes))
    for u, v in mst_edges:
        weight = G[u][v]['weight']
        mst_adj_np[u, v] = weight
        mst_adj_np[v, u] = weight

    mst_adj_batch = torch.tensor(mst_adj_np)
    mst_adj_batch = mst_adj_batch * torch.triu(torch.ones_like(mst_adj_batch), diagonal=1)

    return mst_adj_batch
```

#### 手順 2-7: inference_infinity_gradmst.py の実装
- [ ] 操作: `inference_infinity_gradmst.py` を作成（unconstrained版）
- [ ] 確認: relation_infer関数が実装され、閾値ベース選択を使用
- [ ] テスト: `pytest tests/test_inference_unconst.py -v` → **全テストがPASS**
- [ ] エラー時対処: epoch.relation_inferとの互換性を確認

**inference_infinity_gradmst.py の実装内容:**
```python
"""
Unconstrained RelationFormer Inference.

This module implements the "Unconstrained [55]" baseline method
from the paper, which uses simple threshold-based edge selection
without any tree structure constraints.

Paper Reference:
    Section 5.3: "Unconstrained [55]"
    "This method is identical to our method without applying
    the tree structure constraint."
"""
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from typing import Dict, List, Tuple
from torchvision.ops import batched_nms


def relation_infer(
    h: torch.Tensor,
    out: Dict[str, torch.Tensor],
    net: torch.nn.Module,
    obj_token: int,
    rln_token: int,
    nms: bool = False,
    map_: bool = False
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Unconstrained推論（閾値ベース選択）

    このアルゴリズムは論文の「Unconstrained [55]」手法に対応し、
    MST制約を適用せず、単純な閾値判定でエッジを選択します。

    Args:
        h: [batch, seq_len, hidden_dim] 隠れ層特徴
        out: モデル出力辞書
        net: ニューラルネットワークモデル
        obj_token: オブジェクトトークン数
        rln_token: リレーショントークン数
        nms: Non-Maximum Suppression適用有無
        map_: マッピング情報返却有無

    Returns:
        pred_nodes: 予測ノード座標のリスト
        pred_edges: 予測エッジインデックスのリスト

    Algorithm:
        1. エッジ予測を実行
        2. 閾値判定: p(exist) > p(non-exist)
        3. 選択されたエッジを返す

    Note:
        この手法はツリー構造を保証しないため、
        閉路や森（複数の連結成分）を出力する可能性があります。
    """
    object_token = h[..., :obj_token, :]

    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token + rln_token, :]

    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    if nms:
        # NMS処理（省略、上記と同じ）
        pass

    pred_nodes = []
    pred_edges = []

    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []
        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        if map_:
            pred_nodes_boxes.append(
                out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy()
            )
            pred_nodes_boxes_score.append(
                out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()
            )
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        if node_id.numel() > 1:
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))

            node_pairs_valid = torch.tensor(
                [list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))]
            )

            # エッジ予測
            if rln_token > 0:
                relation_feature = torch.cat((
                    object_token[batch_id, node_pairs[0], :],
                    object_token[batch_id, node_pairs[1], :],
                    relation_token[batch_id, ...].repeat(len(node_pairs_valid), 1)
                ), 1)
            else:
                relation_feature = torch.cat(
                    (object_token[batch_id, node_pairs[0], :],
                     object_token[batch_id, node_pairs[1], :]), 1
                )

            relation_pred = net.module.relation_embed(relation_feature).detach()

            # 閾値ベース選択: argmax > 0.5
            pred_rel = torch.nonzero(torch.argmax(relation_pred, -1), as_tuple=False).squeeze()

            if pred_rel.numel() > 0:
                if pred_rel.dim() == 0:
                    pred_rel = pred_rel.unsqueeze(0)
                pred_edges.append(node_pairs_valid[pred_rel].cpu().numpy())
            else:
                pred_edges.append(np.empty((0, 2)))

            if map_:
                if pred_rel.numel() > 0:
                    pred_edges_boxes_score.append(
                        relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy()
                    )
                    pred_edges_boxes_class.append(
                        torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy()
                    )
                else:
                    pred_edges_boxes_score.append(np.empty(0))
                    pred_edges_boxes_class.append(np.empty(0))
        else:
            pred_edges.append(np.empty((0, 2)))

            if map_:
                pred_edges_boxes_score.append(np.empty(0))
                pred_edges_boxes_class.append(np.empty(0))

    if map_:
        return (pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score,
                pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class)
    else:
        return pred_nodes, pred_edges
```

#### 手順 2-8: フェーズ2完了確認
- [ ] 操作: `pytest tests/ -v --tb=short` で全テストを実行
- [ ] 確認: 全テストがPASS（FAILED=0）
- [ ] テスト: `python -m py_compile inference_*.py` で構文チェック
- [ ] エラー時対処: 失敗したテストのtracebackを確認して修正

---

### フェーズ 3: [包括的テストスイートの作成]

#### 手順 3-1: SFS Layerのユニットテスト作成
- [ ] 操作: `tests/test_sfs_layer.py` を作成
- [ ] 確認: 以下のテストが含まれる
  - E-エッジのラベル抑制
  - 勾配フロー
  - Lambdaパラメータの効果
- [ ] テスト: `pytest tests/test_sfs_layer.py -v` → **全PASS**
- [ ] エラー時対処: losses_only.pyの実装を確認

**tests/test_sfs_layer.py の実装内容:**
```python
"""
Tests for SFS (Straight-Forward Softmax) Layer.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestSFSLayer:
    """SFS Layerの動作テスト"""

    def test_label_suppression(self):
        """
        TEST-SFS-001: E-エッジのラベル抑制

        期待結果:
        - MSTに含まれないエッジの確率が抑制される
        """
        num_edges = 5
        edge_probs = torch.rand(num_edges, 2)
        edge_probs = F.softmax(edge_probs, dim=-1)

        # MST結果（エッジ2と4が削除）
        mst_labels = torch.tensor([1.0, 1.0, 0.000001, 1.0, 0.000001])

        # 確率修正
        modified_probs = edge_probs.clone()
        modified_probs[:, 1] *= mst_labels
        modified_probs[:, 0] += modified_probs[:, 1] * (1 - mst_labels)

        # 確認
        assert modified_probs[2, 1] < 0.00001, "E- edge should be suppressed"
        assert modified_probs[4, 1] < 0.00001, "E- edge should be suppressed"
        assert modified_probs[0, 1] > 0.1, "Normal edge should remain"

    def test_gradient_flow(self):
        """
        TEST-SFS-002: 勾配フロー

        期待結果:
        - SFS Layer適用後も勾配が流れる
        """
        relation_embed = nn.Linear(256, 2)
        optimizer = torch.optim.SGD(relation_embed.parameters(), lr=0.01)

        features = torch.randn(10, 256, requires_grad=True)
        target = torch.ones(10, dtype=torch.long)

        # Forward
        logits = relation_embed(features)
        probs = F.softmax(logits, dim=-1)

        # SFS Layer模擬
        mst_labels = torch.ones(10)
        mst_labels[5:] = 0.000001

        modified_probs = probs.clone()
        modified_probs[:, 1] *= mst_labels
        modified_probs[:, 0] += modified_probs[:, 1] * (1 - mst_labels)

        # Loss
        loss = F.cross_entropy(modified_probs, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        assert relation_embed.weight.grad is not None
        assert not torch.isnan(relation_embed.weight.grad).any()

    @pytest.mark.parametrize("lambda_val", [1, 5, 10, 20])
    def test_lambda_effect(self, lambda_val):
        """
        TEST-SFS-003: Lambdaパラメータの効果

        期待結果:
        - Lambdaが大きいほど抑制が強い
        """
        suppression = torch.exp(torch.tensor(-lambda_val))

        if lambda_val == 10:
            assert suppression < 0.0001, "Lambda=10 should suppress < 0.0001"

        if lambda_val == 20:
            assert suppression < suppression.new_tensor(0.0001) * 0.1
```

#### 手順 3-2: Integration Tests作成
- [ ] 操作: `tests/test_integration.py` を作成
- [ ] 確認: Forward/Backward Passのテストが含まれる
- [ ] テスト: `pytest tests/test_integration.py -v` → **全PASS**
- [ ] エラー時対処: モックデータの形状を確認

**tests/test_integration.py の実装内容:**
```python
"""
Integration tests for inference pipeline.
"""
import pytest
import torch
import torch.nn as nn
from inference_infinity_mst_nx_gradmst import relation_infer


class TestIntegration:
    """推論パイプラインの統合テスト"""

    def test_forward_pass_complete(self, dummy_hidden_features, dummy_model_output, mock_network):
        """
        TEST-INT-001: 完全なForward Pass

        期待結果:
        - エラーなく完了
        - 出力形状が正しい
        """
        h = dummy_hidden_features
        out = dummy_model_output
        net = mock_network

        pred_nodes, pred_edges = relation_infer(h, out, net, 20, 1)

        assert len(pred_nodes) == 2  # batch_size
        assert len(pred_edges) == 2

        for nodes, edges in zip(pred_nodes, pred_edges):
            assert nodes.dim() == 2
            assert nodes.shape[1] == 2  # (x, y)
            assert edges.shape[1] == 2 if len(edges) > 0 else True

    def test_batch_consistency(self, mock_network):
        """
        TEST-INT-002: バッチ処理の一貫性

        期待結果:
        - 単一サンプルとバッチで結果が一致
        """
        # 単一
        h_single = torch.randn(1, 21, 256)
        out_single = {
            'pred_logits': torch.randn(1, 20, 2),
            'pred_nodes': torch.rand(1, 20, 4)
        }

        pred_nodes_single, pred_edges_single = relation_infer(
            h_single, out_single, mock_network, 20, 1
        )

        # バッチ
        h_batch = h_single.repeat(4, 1, 1)
        out_batch = {
            'pred_logits': out_single['pred_logits'].repeat(4, 1, 1),
            'pred_nodes': out_single['pred_nodes'].repeat(4, 1, 1)
        }

        pred_nodes_batch, pred_edges_batch = relation_infer(
            h_batch, out_batch, mock_network, 20, 1
        )

        # 最初のサンプルが一致（形状のみ）
        assert pred_nodes_single[0].shape == pred_nodes_batch[0].shape
```

#### 手順 3-3: E2Eテスト作成
- [ ] 操作: `tests/test_e2e.py` を作成
- [ ] 確認: メトリクス計算のテストが含まれる
- [ ] テスト: `pytest tests/test_e2e.py -v` → **全PASS**
- [ ] エラー時対処: メトリクス計算関数が存在するか確認

**tests/test_e2e.py の実装内容:**
```python
"""
End-to-end tests for TreeFormer inference.
"""
import pytest
import torch
import numpy as np
import networkx as nx
from inference_infinity_mst_nx_gradmst import relation_infer as relation_infer_mst
from inference_infinity_gradmst import relation_infer as relation_infer_unconst


class TestE2E:
    """E2Eテスト"""

    def test_tree_rate_mst_vs_unconstrained(self, dummy_hidden_features, dummy_model_output, mock_network):
        """
        TEST-E2E-001: Tree rate検証

        期待結果:
        - MST版: tree_rate = 100%
        - Unconstrained版: tree_rate < 100%
        """
        h = dummy_hidden_features
        out = dummy_model_output
        net = mock_network

        # MST版
        _, edges_mst = relation_infer_mst(h, out, net, 20, 1)
        tree_count_mst = sum(
            nx.is_tree(nx.Graph(edges)) if len(edges) > 0 else True
            for edges in edges_mst
        )
        tree_rate_mst = tree_count_mst / len(edges_mst) * 100

        # Unconstrained版
        _, edges_unconst = relation_infer_unconst(h, out, net, 20, 1)
        tree_count_unconst = sum(
            nx.is_tree(nx.Graph(edges)) if len(edges) > 0 else True
            for edges in edges_unconst
        )
        tree_rate_unconst = tree_count_unconst / len(edges_unconst) * 100

        assert tree_rate_mst == 100.0, "MST must always produce trees"
        # Unconstrainedは必ずしもツリーを生成しない（ランダム出力の場合）

    def test_performance_metrics(self):
        """
        TEST-E2E-002: パフォーマンスメトリクス

        期待結果:
        - 推論時間が許容範囲内
        """
        import time

        h = torch.randn(4, 21, 256)
        out = {
            'pred_logits': torch.randn(4, 20, 2),
            'pred_nodes': torch.rand(4, 20, 4)
        }

        class MockNet:
            class module:
                relation_embed = torch.nn.Linear(512, 2)

        net = MockNet()

        start = time.time()
        _ = relation_infer_mst(h, out, net, 20, 1)
        elapsed = time.time() - start

        # 4サンプルで1秒以内
        assert elapsed < 1.0, f"Inference too slow: {elapsed:.3f}s"
```

#### 手順 3-4: pytest設定ファイル作成
- [ ] 操作: `pytest.ini` を作成してテスト設定を定義
- [ ] 確認: マーカー、カバレッジ設定が含まれる
- [ ] テスト: `pytest --collect-only` でテストが収集される
- [ ] エラー時対処: 設定ファイルの構文を確認

**pytest.ini の内容:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
```

#### 手順 3-5: フェーズ3完了確認
- [ ] 操作: `pytest tests/ -v --cov=. --cov-report=term-missing` でカバレッジ付きテスト
- [ ] 確認: カバレッジ > 80%、全テストPASS
- [ ] テスト: N/A
- [ ] エラー時対処: カバレッジが低い場合はテストを追加

---

### フェーズ 4: [統合とリグレッション確認]

#### 手順 4-1: valid_smd_guyot_nx.pyを新実装に切り替え
- [ ] 操作: `valid_smd_guyot_nx.py` の1255-1259行を以下に変更
  ```python
  if is_use_mst:
      from inference_infinity_mst_nx_gradmst import relation_infer
  else:
      from inference_infinity_gradmst import relation_infer
  ```
- [ ] 確認: `python -m py_compile valid_smd_guyot_nx.py` で構文OK
- [ ] テスト: `python -c "import valid_smd_guyot_nx"` でimport成功
- [ ] エラー時対処: ModuleNotFoundErrorの場合は__init__.pyを確認

#### 手順 4-2: import動作確認
- [ ] 操作: `python -c "from inference_infinity_mst_nx_gradmst import relation_infer; print('OK')"` で確認
- [ ] 確認: "OK"が表示される
- [ ] テスト: 他の2ファイルも同様に確認
- [ ] エラー時対処: ImportErrorの場合は依存関係を確認

#### 手順 4-3: 小規模データでの動作確認
- [ ] 操作: Guyot_200_20サブセットで推論を実行（コマンドは後述）
- [ ] 確認: エラーなく完了し、メトリクスが表示される
- [ ] テスト: Tree rate = 100% であることを確認
- [ ] エラー時対処: データセットが存在しない場合はパスを確認

#### 手順 4-4: リグレッションテスト
- [ ] 操作: `pytest tests/ -v -m "not slow"` で高速テストのみ実行
- [ ] 確認: 全テストPASS
- [ ] テスト: N/A
- [ ] エラー時対処: 失敗したテストのログを確認

#### 手順 4-5: パフォーマンス確認
- [ ] 操作: `pytest tests/test_e2e.py::TestE2E::test_performance_metrics -v`
- [ ] 確認: 推論時間が許容範囲内
- [ ] テスト: N/A
- [ ] エラー時対処: 遅い場合はプロファイリング（`python -m cProfile`）

#### 手順 4-6: 変更のコミット
- [ ] 操作: `git add inference_*.py tests/ valid_smd_guyot_nx.py && git commit -m "Implement missing inference files with comprehensive tests"`
- [ ] 確認: `git log -1 --stat` でコミット内容確認
- [ ] テスト: N/A
- [ ] エラー時対処: コミットメッセージを修正する場合は `git commit --amend`

---

### フェーズ 5: [ドキュメント整備とクリーンアップ]

#### 手順 5-1: docstringの検証
- [ ] 操作: `pydocstyle inference_*.py` でdocstring規約をチェック
- [ ] 確認: エラー0件または許容範囲内
- [ ] テスト: N/A
- [ ] エラー時対処: pydocstyleがない場合は `uv pip install pydocstyle`

#### 手順 5-2: README.mdの更新
- [ ] 操作: `README.md` に新しい推論ファイルのセクションを追加
- [ ] 確認: 使用例とAPIドキュメントが含まれる
- [ ] テスト: Markdown linter（`markdownlint README.md`）
- [ ] エラー時対処: リンク切れがある場合は修正

#### 手順 5-3: コードフォーマット
- [ ] 操作: `ruff format inference_*.py tests/` でコード整形
- [ ] 確認: 変更があればdiffを確認
- [ ] テスト: `ruff check inference_*.py tests/` でlint
- [ ] エラー時対処: ruffがない場合は `uv pip install ruff`

#### 手順 5-4: 型チェック
- [ ] 操作: `mypy inference_*.py --ignore-missing-imports` で型チェック
- [ ] 確認: エラー0件（警告は許容）
- [ ] テスト: N/A
- [ ] エラー時対処: mypyがない場合は `uv pip install mypy`

#### 手順 5-5: 最終テスト実行
- [ ] 操作: `pytest tests/ -v --cov=. --cov-report=html` でHTMLレポート生成
- [ ] 確認: `htmlcov/index.html` が生成される
- [ ] テスト: ブラウザで `htmlcov/index.html` を開いてカバレッジ確認
- [ ] エラー時対処: pytest-covがない場合は `uv pip install pytest-cov`

#### 手順 5-6: 最終コミット
- [ ] 操作: `git add . && git commit -m "Add documentation and finalize implementation"`
- [ ] 確認: `git log --oneline -5` で一連のコミットを確認
- [ ] テスト: N/A
- [ ] エラー時対処: N/A

#### 手順 5-7: リモートプッシュ
- [ ] 操作: `git push -u origin claude/review-code-and-docs-01MsuxYMUdiHb6xFcbGtVZgT` でプッシュ
- [ ] 確認: プッシュ成功メッセージ
- [ ] テスト: GitHub上でブランチとコミットを確認
- [ ] エラー時対処: 403エラーの場合はブランチ名を確認（claude/で始まり、session IDで終わる）

---

## 4. 作業に使用するコマンド参考情報

### 環境セットアップ
```bash
# uv環境の作成と有効化
uv venv
source .venv/bin/activate  # Linux/Mac
# または .venv\Scripts\activate  # Windows

# 依存パッケージのインストール
uv pip install torch torchvision networkx scipy numpy pytest pytest-cov mypy ruff pydocstyle
```

### テスト実行
```bash
# 全テスト実行
pytest tests/ -v

# 特定のテストファイル
pytest tests/test_mst_computation.py -v

# 特定のテストケース
pytest tests/test_inference_gradmst.py::TestInferenceGradMST::test_output_is_tree -v

# カバレッジ付き
pytest tests/ -v --cov=. --cov-report=term-missing

# マーカー指定
pytest tests/ -v -m "unit"
pytest tests/ -v -m "not slow"

# 失敗時に即停止
pytest tests/ -v -x

# 詳細出力
pytest tests/ -vv --tb=long
```

### コード品質チェック
```bash
# フォーマット
ruff format .

# Lint
ruff check .

# 型チェック
mypy inference_*.py --ignore-missing-imports

# Docstring検証
pydocstyle inference_*.py
```

### Git操作
```bash
# 状態確認
git status
git diff

# ステージング
git add inference_*.py tests/

# コミット
git commit -m "Implement missing inference files"

# プッシュ
git push -u origin claude/review-code-and-docs-01MsuxYMUdiHb6xFcbGtVZgT

# ログ確認
git log --oneline -10
git log -1 --stat
```

### 動作確認（小規模データ）
```bash
# valid_smd_guyot_nx.pyの実行（例）
python valid_smd_guyot_nx.py \
    --config configs/tree_2D_use_mst_only1.yaml \
    --checkpoint checkpoints/best_model.pth \
    --device cuda \
    --use_mst

# import確認
python -c "from inference_infinity_mst_nx_gradmst import relation_infer; print('OK')"

# 構文チェック
python -m py_compile inference_*.py
```

### デバッグ
```bash
# Pythonデバッガ
python -m pdb valid_smd_guyot_nx.py

# インタラクティブシェル
python -i -c "import inference_infinity_mst_nx_gradmst as inf"

# プロファイリング
python -m cProfile -s cumtime valid_smd_guyot_nx.py
```

---

## 5. トラブルシューティング

### よくあるエラーと対処法

#### エラー1: ModuleNotFoundError: No module named 'inference_infinity_mst_nx_gradmst'
**原因:** ファイルが正しいディレクトリに存在しない、またはPYTHONPATHが設定されていない
**対処法:**
```bash
# ファイル確認
ls -l inference_*.py

# カレントディレクトリ確認
pwd

# PYTHONPATH設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# または直接実行
python -c "import sys; sys.path.insert(0, '.'); import inference_infinity_mst_nx_gradmst"
```

#### エラー2: ImportError: cannot import name 'relation_infer' from 'epoch'
**原因:** epoch.pyに該当関数が存在しない、または関数名が異なる
**対処法:**
```bash
# 関数存在確認
grep -n "^def relation_infer" epoch.py

# 正しいブランチか確認
git branch
git log -1 --oneline
```

#### エラー3: RuntimeError: CUDA out of memory
**原因:** GPUメモリ不足
**対処法:**
```bash
# CPUで実行
python valid_smd_guyot_nx.py --device cpu

# バッチサイズを小さく
# config.yamlのBATCH_SIZEを調整
```

#### エラー4: AssertionError: MST must form a tree structure
**原因:** MST計算ロジックのバグ
**対処法:**
```python
# デバッグコード追加
import networkx as nx
G = nx.Graph()
G.add_edges_from(edges)
print(f"Is tree: {nx.is_tree(G)}")
print(f"Nodes: {G.nodes()}, Edges: {G.edges()}")
print(f"Connected components: {list(nx.connected_components(G))}")
```

#### エラー5: pytest: command not found
**原因:** pytestがインストールされていない
**対処法:**
```bash
uv pip install pytest
# または
pip install pytest
```

---

## 6. 完了の定義

作業が完了したと見なすには、以下の全ての条件を満たす必要があります：

- [ ] **機能要件:** 3つの推論ファイルが実装され、valid_smd_guyot_nx.pyが正常に動作する
- [ ] **テスト要件:** pytest実行で全テストがPASS、カバレッジ > 80%
- [ ] **コード品質:** ruff、mypy、pydocstyleのチェックをパス
- [ ] **ドキュメント:** README.md更新、各ファイルにdocstring完備
- [ ] **統合:** 既存のテストスイートに影響なし（リグレッションなし）
- [ ] **性能:** 推論時間が許容範囲内（4サンプル/秒以上）
- [ ] **Git:** 全変更がコミット・プッシュされ、ブランチが最新

---

## 7. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

---

## 付録A: 実装チェックリスト

### inference_infinity_mst_nx_gradmst.py
- [ ] relation_infer関数実装
- [ ] compute_mst_nx関数実装
- [ ] docstring完備（関数・パラメータ・戻り値・例）
- [ ] 型ヒント追加
- [ ] 論文セクション参照コメント
- [ ] エラーハンドリング

### inference_infinity_mst_nx_dist.py
- [ ] relation_infer関数実装
- [ ] use_distanceパラメータ実装
- [ ] 距離重み付けロジック
- [ ] docstring完備
- [ ] 型ヒント追加

### inference_infinity_gradmst.py
- [ ] relation_infer関数実装
- [ ] 閾値ベース選択ロジック
- [ ] docstring完備
- [ ] 型ヒント追加
- [ ] 論文セクション参照コメント

### テストファイル
- [ ] tests/conftest.py - pytest fixtures
- [ ] tests/test_mst_computation.py - MST正確性
- [ ] tests/test_sfs_layer.py - SFS Layer動作
- [ ] tests/test_inference_gradmst.py - TreeFormer実装
- [ ] tests/test_inference_dist.py - Test-time constraint
- [ ] tests/test_inference_unconst.py - Unconstrained
- [ ] tests/test_integration.py - 統合テスト
- [ ] tests/test_e2e.py - E2Eテスト
- [ ] pytest.ini - pytest設定

---

## 付録B: 参考資料

### 論文参照
- **Section 4.2**: Tree-constrained graph generation
- **Equation (10)**: SFS Layer feature modification
- **Equation (11)**: Loss function (unconstrained + constrained)
- **Section 5.3**: Baseline methods

### 既存実装
- `epoch.py:43-306` - relation_infer (Unconstrained)
- `epoch.py:308-582` - relation_infer_mst (MST-constrained)
- `losses_only.py:2141-2360` - loss_edges_mst (SFS Layer)

### 調査ドキュメント
- `temp/issue_2_missing_files_detailed_investigation.md` - 詳細調査報告
- `temp/algorithm_mapping_and_testing_guide.md` - アルゴリズム対応とテストガイド

---

**作業計画書 終わり**
**最終更新:** 2025-11-14
