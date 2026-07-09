# PR #2 レビュー・選別取り込み 引継書

作成日: 2026-07-09

## 目的

GitHub PR #2 `claude/implement-missing-inference-files-01D32vfvA14H5qmVAMvjgmhS` を直接 merge せず、フォーク元 `huntorochi/TreeFormer` と TreeFormer 論文の手法を確認しながら、現行 `main` に必要かつ妥当なコードだけを取り込んだ。

## 参照した根拠

- 論文: TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation
  - arXiv: https://arxiv.org/abs/2411.16132
  - 論文上の要点: RelationFormer の edge prediction に対し、edge non-existence probability を MST cost として使い、tree constraint をかける。test-time constraint baseline も論文中で扱われる。
- フォーク元: `huntorochi/TreeFormer`
  - default branch: `main`
  - 先端確認時点: `8c2679f Modify training epochs and log experiment name`
- 作業対象 repo: `/home/kasm-user/Desktop/TreeFormer`

## PR #2 コミット別レビュー結果

PR #2 のコミット列:

1. `534a191 Add comprehensive investigation of missing inference files`
   - ドキュメント調査のみ。取り込み対象外。
2. `bf45f9b Add comprehensive work plan for missing inference files implementation`
   - 作業計画のみ。取り込み対象外。
3. `6c26b98 Update work plan to support parallel execution with subagents`
   - 作業計画更新のみ。取り込み対象外。
4. `539910b Fix: Use existing epoch.py functions for inference (temporary)`
   - 一時対応。最終実装では不要。
5. `131f7ca feat: Implement 3 missing inference files`
   - 本質的な取り込み対象。ただし生コードは `net.module.relation_embed` 固定で、現行 `valid_smd_guyot_nx.py` の plain model 呼び出しと不整合。
   - そのまま取り込まず、同等ロジックを current main 向けに整理して実装。
6. `fd40a2f test: Add pytest fixtures and test infrastructure`
   - `tests/conftest.py` 追加。現行テストには不要。取り込み対象外。
7. `c0643e9 fix: Switch to implemented inference files`
   - `valid_smd_guyot_nx.py` の import 切替。現行 `origin/main` はすでに同じ import 名を参照していたため変更不要。
8. `bd94ba7 docs: Update work plan with progress through Phase 4-2`
   - ドキュメントのみ。取り込み対象外。
9. `98918d5 docs: Add final git push entry to work log`
   - ドキュメントのみ。取り込み対象外。
10. `9cc4434 chore: Add .gitignore for Python and project files`
    - 現行 main の `.gitignore` と今回の scope に照らして不要。取り込み対象外。
11. `905ffcb docs: Add training verification work plan`
    - ドキュメントのみ。取り込み対象外。
12. `a9be8c2 chore: Remove old work plan file`
    - ドキュメント整理のみ。取り込み対象外。
13. `cc3cd23 docs: Add comprehensive onboarding guide`
    - ドキュメントのみ。取り込み対象外。

## 実装方針

フォーク元の `inference_infinity.py` / `inference_infinity_mst_nx.py` は直接取り込まなかった。理由は、PR #2 の目的が current main の `valid_smd_guyot_nx.py` が参照する欠落ファイルを補うことであり、フォーク元ファイルを戻すと current main の Guyot pipeline と衝突する差分が大きいため。

代わりに以下を追加:

- `inference_treeformer.py`
  - 共通 helper。
  - plain model と DDP/DataParallel の両方に対応する `get_relation_embed()` を実装。
  - unconstrained inference と MST inference を同じ return contract で扱う。
  - MST は NetworkX Kruskal を使用し、論文の `p(non-edge)` cost に沿う。
  - PR #2 生コードにあった zero-cost MST edge が `torch.nonzero` で落ちる問題を避けるため、MST edge list を直接返す。
- `inference_infinity_gradmst.py`
  - PR #2 の unconstrained inference 名を満たす wrapper。
- `inference_infinity_mst_nx_gradmst.py`
  - PR #2 の MST inference 名を満たす wrapper。
- `inference_infinity_mst_nx_dist.py`
  - optional distance-weighted MST wrapper。
- `tests/test_inference_treeformer.py`
  - unconstrained edge selection、MST edge selection、zero-cost MST edge、distance option の最小テスト。

## 取り込まなかったもの

- `data/` 配下の画像・annotation dataset。
- `temp/` 配下の大量ドキュメント。
- `tests/__pycache__/*.pyc`。
- PR #2 の `tests/conftest.py`。
- フォーク元の `inference_infinity.py` / `inference_infinity_mst_nx.py`。
- フォーク元の training config 変更。

## 検証済みコマンド

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/kasm-user/Desktop/venv/TreeFormer/bin/python -m pytest -p no:cacheprovider tests/test_inference_treeformer.py tests/test_guyot_dataset.py -q
```

結果:

```text
8 passed, 2 warnings
```

```bash
/home/kasm-user/Desktop/venv/TreeFormer/bin/python -m ruff check inference_treeformer.py inference_infinity_gradmst.py inference_infinity_mst_nx_gradmst.py inference_infinity_mst_nx_dist.py tests/test_inference_treeformer.py
```

結果:

```text
All checks passed!
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/kasm-user/Desktop/venv/TreeFormer/bin/python - <<'PY'
import valid_smd_guyot_nx
print('valid_smd_guyot_nx import ok')
PY
```

結果:

```text
valid_smd_guyot_nx import ok
```

## 既存セットアップ差分

PR #2 取り込みとは別に、先行作業で `uv` / CUDA 11.8 環境用の差分がある。

- `pyproject.toml`
  - `setuptools<81`
  - Linux では PyTorch / torchvision を cu118 index から解決
  - `mmcv` build 用 extra build dependency
- `uv.lock`

CUDA ops は venv に手動 install 済み。`uv sync` を再実行すると `MultiScaleDeformableAttention` が削除対象になる可能性があるため、その場合は再ビルドが必要。

再ビルドコマンド:

```bash
cd /home/kasm-user/Desktop/TreeFormer/models/ops
PATH=/usr/local/cuda/bin:$PATH /home/kasm-user/Desktop/venv/TreeFormer/bin/python setup.py build install
```

## 残リスク・次アクション

- 実 checkpoint / dataset を使った `valid_smd_guyot_nx.py` の end-to-end 評価は未実施。
- distance-weighted MST は wrapper と単体テストのみ。実験上採用する場合は config/CLI の明示導線を別途設計する。
- `valid_smd_guyot_nx.py` は現在 `inference_infinity_mst_nx_gradmst` を default にしている。論文比較のため unconstrained / test-time constraint / distance variant を切り替える CLI が必要なら別作業で追加する。
