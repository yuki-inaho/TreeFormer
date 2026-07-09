# LLMオンボーディングサマリー

> このドキュメントは、新任LLMエージェントが TreeFormer / Guyot training pipeline の現状を短時間で把握し、破壊的な変更を避けながら作業を開始するための初期資料です。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation。単一視点の植物画像から 2D tree graph skeleton を推定する研究実装。
- **最終成果物:** raw 3D2cut Single Guyot dataset から TreeFormer の既存 training loop へ接続し、Guyot 向けの訓練・smoke 検証を再現可能にすること。
- **ビジネス背景・価値:** 植物形状の skeleton graph 推定を自動化し、剪定・計測・認識パイプラインへの応用可能性を高める。
- **現時点の進捗サマリ:** uv 環境、CUDA operator build、pretrained checkpoint 読み込み、raw Guyot dataset loader、training adapter、dry-run、1-step smoke training まで完了。full training は最新方針では未実施で、任意の後続作業。

## 2. クリティカルな要求・制約

> 「壊してはいけない」品質・仕様ラインを箇条書きで列挙します。

- dataset / checkpoint / smoke training output は repo 外、主に `/home/kasm-user/Desktop/TreeFormer_assets/` に置く。Git に混入させない。
- private dataset の元パス、生成スクリプト名、収集条件、内部ラベル名は docs / workdoc に書かない。公開可能な範囲は、TreeFormer が消費する汎用フォーマット、split 構造、ファイル命名規則までに留める。
- 既存の legacy training loop と `custom_collate_fn` の 9-tuple contract を壊さない。
- raw `GuyotDataset` は `{image, nodes, edges, filename}` の dict contract を維持し、training 接続は `GuyotTrainingAdapter` で行う。
- `DATA.DATA_PATH` / `TRAIN.SAVE_PATH` などの config path を優先し、古い hard-coded dataset path に戻さない。
- 学習入力は現行 TreeFormer と同様に RGB 画像を主入力とする。RGB-D 由来データであっても、TreeFormer 側の dataset には RGB 画像と 2D graph annotation を渡す。
- smoke training は完了済みだが、full training の完走は未保証。full training を開始する場合は別途実行計画とログ保存方針を決める。

## 3. 参照すべき合意済み資料

> 新任エージェントが必ず確認すべき一次資料の一覧です。パスと役割を記載します。

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| Upstream README | `README.md` | 元実装の概要、dataset 構造、training / evaluation コマンドの前提 |
| 作業計画・記録 | `temp/workdoc_Jul08-2026_treeformer_guyot_pipeline.md` | 実施済み作業、検証ログ、DoD、scope 変更、監査結果 |
| Guyot loader | `guyot_dataset.py` | raw 3D2cut annotation parser、dataset、training adapter |
| smoke config | `configs/tree_2D_guyot_train_smoke.yaml` | 1-step smoke training の正本 config |
| dry-run config | `configs/tree_2D_guyot_dry_run.yaml` | model / dataloader 初期化確認用 config |
| full train config | `configs/tree_2D_guyot_full_train.yaml` | 任意後続の full training 用 config。未実行 |
| テスト資産 | `tests/test_guyot_dataset.py` | parser、dataset、adapter contract の pytest |
| 既知課題リスト | `temp/workdoc_Jul08-2026_treeformer_guyot_pipeline.md` | residual risk、未実施事項、lint分類を参照 |
| pretrained weights | `/home/kasm-user/Desktop/TreeFormer_assets/pretrained_weights/fork_source_main/` | フォーク元 README の Google Drive から取得した checkpoint。repo 外管理 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- Guyot loader / adapter の contract を保った小さな修正。
- smoke training config、dry-run config、pytest の保守。
- full training を開始する前の事前検証、ログ設計、checkpoint 保存先確認。
- README / docs / workdoc への実行手順追記。

### 任せないタスク（例）

- 明示指示なしの full training 長時間実行。
- repo 内への dataset / checkpoint / `.tar.gz` / `.pkl` / `.npz` 追加。
- private dataset の実パス、生成コマンド、内部由来が分かる名前の docs 追記。
- 既存 training loop、loss、model architecture の大規模変更。
- user / 他エージェント由来の未追跡ファイルの勝手な削除や取り込み。

## 5. インタラクション方針

- **回答スタイル:** 日本語、簡潔、見出し＋箇条書き中心。実行結果は command、exit status、主要ログを添える。
- **回答手順:** 前提確認 → 変更範囲 → 実行コマンド → 検証結果 → 残課題の順で報告する。
- **禁止事項・注意:** 未実行の full training を完了済みと書かない。推測を事実として断定しない。dataset/checkpoint の Git 混入を放置しない。
- **秘匿情報の扱い:** GitHub token、ローカル認証情報、外部サービス認証情報、private dataset の取得 URL や資格情報は文書化しない。

## 6. 試行タスク（オンボーディング演習）

> 小さな検証タスクを2〜3件記載してください。理解度を確認するために実施します。

1. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/kasm-user/Desktop/venv/TreeFormer/bin/python -m pytest -p no:cacheprovider tests/test_guyot_dataset.py -q` を実行し、parser / dataset / adapter contract が通ることを確認する。
2. `configs/tree_2D_guyot_train_smoke.yaml` を読み、`DATA.TRAIN_LIMIT: 1`、`DATA.VAL_LIMIT: 1`、repo 外 `TRAIN.SAVE_PATH` が設定されていることを説明する。
3. `git status --ignored --short` で checkpoint / dataset / cache が repo に混入していないことを確認し、残る ignored artifact が CUDA build artifact だけかを報告する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 変更した理由、対象ファイル、検証コマンド、結果、残課題を同じ更新に含める。
- **TBDの扱い:** TBD は owner、確認方法、期限または次アクションを併記する。単独の TBD を残さない。
- **レビュー/承認フロー:** 実装 → 最小テスト → artifact 混入確認 → workdoc / docs 更新 → commit。長時間学習は開始前に方針確認する。
- **その他の運用ルール:** `git add -A` は避け、混在 worktree では明示ファイルだけ stage する。生成物 cleanup 後に `git status` を確認する。

## 8. Pretrained Weights

フォーク元 `huntorochi/TreeFormer` の README は pretrained checkpoint を Google Drive folder として公開している。

- source repository: `https://github.com/huntorochi/TreeFormer`
- Google Drive folder: `https://drive.google.com/drive/folders/1QFIwOAESSAF8Uc4it0-cAzBiMMszNJg2?usp=sharing`
- local asset path: `/home/kasm-user/Desktop/TreeFormer_assets/pretrained_weights/fork_source_main/`
- repo には checkpoint を置かない。

取得コマンド:

```bash
mkdir -p /home/kasm-user/Desktop/TreeFormer_assets/pretrained_weights/fork_source_main

uvx --from gdown gdown --folder \
  'https://drive.google.com/drive/folders/1QFIwOAESSAF8Uc4it0-cAzBiMMszNJg2?usp=sharing' \
  -O /home/kasm-user/Desktop/TreeFormer_assets/pretrained_weights/fork_source_main
```

確認済みの配置:

```text
pretrained_weights/fork_source_main/
├── grapevein/
│   ├── checkpoint_ours.pkl
│   └── checkpoint_unmst.pkl
├── root/
│   ├── checkpoint_ours.pkl
│   └── checkpoint_unmst.pkl
└── synthetic/
    ├── checkpoint_ours.pkl
    └── checkpoint_unmst.pkl
```

`checkpoint_ours.pkl` は MST/tree-constrained 系、`checkpoint_unmst.pkl` は unconstrained 系として扱う。実際の評価 config へ接続する場合は、対象 dataset と model config の対応を確認してから `--checkpoint` に渡す。

## 9. Dataset Format

TreeFormer の legacy dataloader は split ごとの dataset root を `LoadCNNDataset(parent_path=...)` に渡す。入力は `img/` 配下の RGB 画像で、同名 stem の `.pt` graph annotation を `data/` から読む。

split root の基本構造:

```text
<split_root>/
├── data/
│   └── <sample_id>.pt
├── img/
│   └── <sample_id>.png
├── check/
│   └── <sample_id>.png
└── unet/
    └── <sample_id>.png
```

`LoadCNNDataset` が直接読む最小要件:

- `data/<sample_id>.pt`
  - `list_DETR_points_left_up`: normalized 2D node coordinates
  - `DETR_node_collections`: graph connectivity / edge path information
- `img/<sample_id>.png`
  - RGB image input。alpha channel がある場合は先頭 3 channel のみ使う。

`check/` と `unet/` は legacy pipeline の補助画像・可視化・派生 mask 用として残す。新規 private dataset の場合も docs に実パスや生成元を残さず、上記の抽象 layout と file contract だけを書く。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:**
  - repo: `/home/kasm-user/Desktop/TreeFormer`
  - assets: `/home/kasm-user/Desktop/TreeFormer_assets`
  - uv venv: `/home/kasm-user/Desktop/venv/TreeFormer`
  - raw Guyot extracted dataset: `/home/kasm-user/Desktop/TreeFormer_assets/datasets/3D2cut_Single_Guyot_extracted`
  - pretrained weights: `/home/kasm-user/Desktop/TreeFormer_assets/pretrained_weights/fork_source_main`
  - smoke outputs: `/home/kasm-user/Desktop/TreeFormer_assets/trained_weights_smoke`
- **代表的なコマンド:**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/kasm-user/Desktop/venv/TreeFormer/bin/python -m pytest -p no:cacheprovider tests/test_guyot_dataset.py -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /home/kasm-user/Desktop/venv/TreeFormer/bin/python -m torch.distributed.run --nproc_per_node=1 --master_port=29532 train_mst.py --config configs/tree_2D_guyot_train_smoke.yaml --cuda_visible_device 0 --local_rank 0

git status --short --branch --untracked-files=all
git status --ignored --short
```

- **依存ライブラリ:** `pyproject.toml` を参照。主要依存は PyTorch 2.6 系、torchvision 0.21 系、MMCV 1.7 系、scikit-image 0.25 系、NetworkX、Pillow、PyYAML。
- **連絡先/責任者:** 未定。GitHub repository owner は `yuki-inaho`。

> ※この文書は現在の smoke training 完了スコープに基づく。full training の実行結果を得た場合は、config、checkpoint path、主要ログ、失敗時の recovery 手順を追記する。
