# LLMオンボーディングサマリー

> このドキュメントは、新任LLMエージェントが TreeFormer / Guyot training pipeline の現状を短時間で把握し、破壊的な変更を避けながら作業を開始するための初期資料です。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation。単一視点の植物画像から 2D tree graph skeleton を推定する研究実装。
- **最終成果物:** raw 3D2cut Single Guyot dataset から TreeFormer の既存 training loop へ接続し、Guyot 向けの訓練・smoke 検証を再現可能にすること。
- **ビジネス背景・価値:** 植物形状の skeleton graph 推定を自動化し、剪定・計測・認識パイプラインへの応用可能性を高める。
- **現時点の進捗サマリ:** uv 環境、CUDA operator build、pretrained checkpoint 読み込み、raw Guyot dataset loader、training adapter、dry-run、1-step smoke training まで完了。full training は repo 外ログ・checkpoint 保存で実行する。

## 2. クリティカルな要求・制約

> 「壊してはいけない」品質・仕様ラインを箇条書きで列挙します。

- dataset / checkpoint / smoke training output は repo 外、原則 `${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}` 配下に置く。Git に混入させない。
- `.env` / `.env.*` / `.agents/` / `temp/` は Git 管理外にする。local path、認証情報、agent 作業メモ、drop 由来ファイルを remote に含めない。
- private dataset の元パス、生成スクリプト名、収集条件、内部ラベル名は docs / workdoc に書かない。公開可能な範囲は、TreeFormer が消費する汎用フォーマット、split 構造、ファイル命名規則までに留める。
- 既存の legacy training loop と `custom_collate_fn` の 9-tuple contract を壊さない。
- raw `GuyotDataset` は `{image, nodes, edges, filename}` の dict contract を維持し、training 接続は `GuyotTrainingAdapter` で行う。
- `DATA.DATA_PATH` / `TRAIN.SAVE_PATH` などの config path を優先し、古い hard-coded dataset path に戻さない。
- 学習入力は現行 TreeFormer と同様に RGB 画像を主入力とする。legacy loader は `ToTensor()` 後に RGB を `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` し、モデル入力を `[0, 1]` から `[-1, 1]` にスケールする。RGB-D 由来データであっても、TreeFormer 側の dataset には RGB 画像と 2D graph annotation を渡す。
- smoke training は完了済みだが、full training の完走は未保証。full training を開始する場合は別途実行計画とログ保存方針を決める。
- private legacy TreeFormer-format dataset で GPU smoke を行う場合は、dataset root を `TREEFORMER_PRIVATE_DATA` 環境変数で渡し、`just cfg-private-pretrained-gpu-batch12` で構成だけ確認してから `just smoke-private-pretrained-gpu-batch12` を実行する。batch size は 12 を初期値とし、smoke recipe は `DATA.TRAIN_LIMIT=24` / `DATA.VAL_LIMIT=12` を明示する。EMA は GPU 上で保持する。
- 現在の private-data curriculum では幾何・変形DAを使わない。Stage 0 は `augmentation=disabled`、Stage 1 以降に使う場合も `augmentation=photometric_opencv` の image-only 光学DAまでに留める。random crop / rotate / scale / affine / perspective / elastic / graph deformation はこの curriculum では使わない。
- graph annotation の粗さで `val/smd` が判断材料にならない場合は、先に dense aux mode を使う。`train=seg_only` は graph decoder output / graph loss / SMD validation をスキップし、RGB から split-local `seg/` の TPE binary mask を背景 + 単一 foreground class の binary loss で直接教師付き学習する。graph 由来 raster mask を segmentation target にしない。評価軸は `val/seg_soft_dice_score`、`val/seg_total_loss`、hard-threshold `val/seg_dice_score` / `val/seg_iou`、`val/pred_positive_rate` とする。
- `train=seg_heatmap` は `seg_only` と同じ外部 mask supervision に、graph annotation 由来の node heatmap target だけを追加する。graph loss は引き続き 0、PAF loss も 0。heatmap loss は `AUX_HEATMAP_MASK_SOURCE=segmentation` と小さい outside weight で foreground 寄りに評価する。前段の segmentation checkpoint から始める場合は `TREEFORMER_PRETRAINED_CHECKPOINT=<best.pt>` と `TREEFORMER_PRETRAINED_KEY=model` を渡す。
- `train=seg_heatmap_paf` は `seg_heatmap` に PAF / edge direction map supervision を追加する。graph output / graph loss は引き続き無効で、`DATA.AUX_TARGET_MODE=seg_heatmap_paf`、`W_AUX_PAF>0`、`AUX_PAF_MASK_SOURCE=paf_and_segmentation` を使う。論文系の direction field を見る段階はこの mode を使い、segmentation-only / heatmap-only の安定性を確認してから進める。
- node heatmap が点ではなく ridge 状に出る場合は、architecture を変える前に `+ablation=heatmap_mse_baseline` / `heatmap_sigma1_5_mse` / `heatmap_focal` / `heatmap_focal_ridge` / `heatmap_focal_ridge_seg_low` で dense aux ablation を行う。比較軸は `val/heatmap_peak_contrast`、`val/masked_heatmap_mae`、`val/seg_iou`、`val/paf_masked_l1`。sigma を変える ablation は cache 生成時の `TREEFORMER_HEATMAP_SIGMA` と training config の `DATA.AUX_HEATMAP_SIGMA` を必ず揃える。
- legacy 互換の `train=seg_supervised` は外部 mask 由来の STDC-style detail boundary を弱い regularizer として使う既存設定。新規実験では、目的が明確な `train=seg_only` / `train=seg_heatmap` / `train=seg_heatmap_paf` を優先する。
- dense aux stage は `FastSegSupervisedDataset` を使う。legacy loader の graph-derived PAF / heatmap 生成を避け、`DATA.AUX_TARGET_MODE` で `seg_only` / `seg_heatmap` / `seg_heatmap_paf` を切り替える。full run 前に repo 外 disk cache を生成し、`seg_only` は `just cache-private-fast-seg`、`seg_heatmap` は `just cache-private-fast-seg-heatmap`、`seg_heatmap_paf` は `just cache-private-fast-seg-heatmap-paf` を使う。cache format v2 は未使用の detail boundary を保存しないため、v1 cache は再利用せず生成し直す。local disk cache の実測では `DATA.NUM_WORKERS=0` が最速だったため、joint graph+aux の pilot / full recipe はこれを既定にする。cache 生成先は `${TREEFORMER_SEG_CACHE_ROOT}` または `${TREEFORMER_ASSETS_ROOT}/cache/fast_seg/...` とし、Git に入れない。
- virtual-root forest graph training は `docs/TreeFormer_virtual_root_forest_formalization.pdf` / `.tex` と `temp/workdoc_Jul09-2026_treeformer_virtual_root_forest.md` の方針に従う。TreeFormer 側は `component_id` などの optional metadata を読み、`train=virtual_root` では metadata 欠落時に暗黙 fallback せず明示エラーにする。
- graph stage で dense aux を併用する場合は `train=joint_virtual_root_aux` を使う。これは virtual-root forest、root head、graph losses、external mask segmentation、detail boundary、node heatmap、PAF / edge-direction supervision を同一 forward で有効にし、`TRAIN.W_JOINT_AUX` で graph loss に対する aux loss の寄与を制御する。`checkpoint.save_last=false` を推奨し、Optuna など多数 trial では `best.pt` と report のみ残す。
- Optuna tuning は任意 dependency group として管理する。導入は `just install-optuna` または `uv pip install --python "$TREEFORMER_PYTHON" --project . --group tuning`。実行前に `just cache-private-joint-virtual-root-aux-optuna` で探索対象のheatmap sigmaごとのcacheを生成し、その後 `just tune-private-joint-virtual-root-aux` を使う。出力はrepo外 `${TREEFORMER_ASSETS_ROOT}/optuna/joint_virtual_root_aux/` のSQLite study、`trials.csv`、日本語 `report.md`、`best_trial_overrides.yaml`。640x480 joint pilot は cache v2、batch 2、worker 0 を安全な既定とし、batchを増やす場合はGPU smokeでVRAMを確認してからoverrideする。report には private dataset 実パスを書かない。
- `FastSegSupervisedDataset` の既定 resize は legacy 互換の `DATA.SEG_RESIZE_POLICY=legacy_half`。raw 800x600 を 512x384 で使う場合は `DATA.MAX_SIZE=512`、生画像 800x600 のまま使う場合は `DATA.MAX_SIZE=800` にし、どちらも `DATA.SEG_RESIZE_POLICY=full` を指定して同じ policy で repo 外 cache を作り直す。
- runtime は Ampere 系 GPU 向けに TF32 と cuDNN benchmark を既定有効にする。segmentation-only full recipe は aux head と aux-loss core の `torch.compile` を既定有効にするが、smoke recipe では初回 compile overhead 回避のため既定無効。whole TreeFormer model compile は custom CUDA deformable attention と Python list/NestedTensor 経路があるため既定で使わない。
- joint graph validation の SMD は既定では legacy CPU backend。必要に応じて任意依存 `geomloss` を `uv pip install --python .venv/bin/python 'geomloss>=0.2.6,<0.3'` で導入し、`TRAIN.SMD_BACKEND=geomloss_gpu` を指定できる。GPU backend は graph post-processing を変えず、point-cloud sampling と regularized transport のみを置換するため、legacy metricとの数値互換性・順位一致を同一checkpointで確認してからcheckpoint選定に使う。
- optional AlbumentationsX backend は `uv pip install --python "$TREEFORMER_PYTHON" --project . --group albumentationsx` で導入する。入れない場合も OpenCV backend で training は継続できる。
- 学習カリキュラムは S0 segmentation-only no-DA で dense mask supervision が学べるかを先に確認し、その後に aux maps、graph no-DA stabilization、必要なら photometric OpenCV へ進む checkpoint-resume 方式を初期案とする。詳細は `docs/HYDRA_TRAINING.md` を参照。
- 学習 stage 完了後は `best.pt` を `infer_panel_treeformer.py` / `just infer-panels` に渡し、validation split の画像ごとに input / ground truth / prediction の summary panel を repo 外へ生成して定性確認する。既定では Hydra checkpoint 内の EMA shadow weights を優先して読む。グラフ stage に aux head がない場合でも、別途 `TREEFORMER_INFER_AUX_CHECKPOINT=<aux_best.pt>` を指定して `just infer-panels-with-aux` を使うと、同じ summary panel に segmentation、node heatmap、edge direction の GT / Pred を診断用に併記できる。
- aux supervised stage 完了後は `infer_aux_panel_treeformer.py` / `just infer-aux-panels` で segmentation overlay、必要に応じた detail boundary head、node heatmap、PAF magnitude/direction の validation panel を repo 外へ生成する。4ch checkpoint では detail boundary は既定で非表示、5ch checkpoint では `Pred detail boundary head` として表示する。segmentation-only checkpoint では loss weight 0 の heatmap / PAF panel は既定で非表示にし、未学習出力を評価対象として読まない。
- batch size 12 の VRAM 目安: RTX A4500 / `DATA.MAX_SIZE=128` / official fork-source `grapevein/checkpoint_ours.pkl` / Muon + ScheduleFree 条件で、1 train batch の既存実測は約 3.1GiB。GPU EMA は model state 約 210MiB を shadow と validation backup に使うため、`ema=default` の運用目安は約 3.5-4.0GiB。8GiB 予算では batch size 12 を初期値としてよい。`nvidia-smi` の GPU 全体使用量は他プロセスを含むため、TreeFormer 単体の VRAM 目安と混同しない。
- CUDA ops の検証は `MultiScaleDeformableAttention` module import と forward double / float check を基準にする。`models/ops/test.py` 全体は high-channel `gradcheck` まで実行するストレステストで、20GB GPU でも OOM し得るため、full test OOM を通常学習 1 batch の OOM と混同しない。

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
| pretrained weights | `${TREEFORMER_ASSETS_ROOT}/pretrained_weights/fork_source_main/` | フォーク元 README の Google Drive から取得した checkpoint。repo 外管理 |
| Hydra training | `docs/HYDRA_TRAINING.md` | Hydra entrypoint、EMA、TensorBoard、checkpoint、Muon + ScheduleFree optimizer の運用 |
| Augmentation module | `treeformer_train/augmentations/` | AlbumentationsX/OpenCV 光学 DA と graph-aware affine / elastic DA。dataset 本体へ直書きしないための composable transform 層 |
| Aux/seg map training | `conf/train/seg_only.yaml`, `conf/train/seg_heatmap.yaml`, `conf/train/seg_heatmap_paf.yaml`, `conf/train/seg_supervised.yaml`, `conf/train/aux_supervised.yaml`, `treeformer_train/aux_training.py`, `treeformer_train/aux_map_targets.py`, `treeformer_train/detail_targets.py` | graph loss を使わず segmentation-only または segmentation + heatmap / PAF を直接 supervised する設定と epoch 実装。detail boundary は外部 mask 由来の任意・弱いaux制約 |
| Dense aux ablations | `conf/ablation/*.yaml`, `summarize_aux_ablation.py` | node heatmap 点状性を比較するための target sigma / focal loss / ridge suppression / seg weight 実験と TensorBoard scalar 集約 |
| Virtual-root forest | `docs/TreeFormer_virtual_root_forest_formalization.pdf`, `docs/TreeFormer_virtual_root_forest_formalization.tex`, `conf/train/virtual_root.yaml`, `treeformer_train/virtual_root.py` | 複数 component の skeleton を偽 bridge で単一木化せず、virtual root 付き forest として扱うための理論資料、config、postprocessor / loss helper |
| Joint graph+aux tuning | `conf/train/joint_virtual_root_aux.yaml`, `treeformer_train/joint_training.py`, `tune_graph_optuna.py` | virtual-root graph stage で segmentation / detail / heatmap / PAF aux を同時に有効化し、Optuna で graph reconstruction 向けに調整する実験基盤 |
| Fast segmentation cache | `treeformer_train/fast_seg_dataset.py`, `generate_fast_seg_cache.py` | segmentation-only stage 用の高速 RGB/mask dataset と repo 外 cache 生成 CLI |
| Runtime speedups | `treeformer_train/runtime.py`, `train_hydra.py`, `conf/config.yaml` | TF32 / cuDNN benchmark / optional aux head and aux loss core `torch.compile`。whole model compile は既定無効 |
| Inference panels | `infer_panel_treeformer.py` | 学習済み checkpoint から画像ごとの input / ground truth / prediction summary panel と graph JSON を repo 外に生成 |
| Aux inference panels | `infer_aux_panel_treeformer.py` | aux-supervised checkpoint から segmentation / optional detail boundary / heatmap / PAF summary panel を repo 外に生成 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク（例）

- Guyot loader / adapter の contract を保った小さな修正。
- smoke training config、dry-run config、pytest の保守。
- augmentation config と transform contract の保守。geometry DA は必ず image と node coordinates を同期する。
- full training を開始する前の事前検証、ログ設計、checkpoint 保存先確認。
- README / docs / workdoc への実行手順追記。
- training stage 後の qualitative check 用 inference panel 生成。checkpoint と dataset root は環境変数または CLI 引数で渡し、出力は `${TREEFORMER_ASSETS_ROOT}` 配下に置く。

### 任せないタスク（例）

- 明示指示なしの full training 長時間実行。
- 画像だけを geometry 変形して graph annotation を更新しない変更。
- repo 内への dataset / checkpoint / `.tar.gz` / `.pkl` / `.npz` 追加。
- private dataset の実パス、生成コマンド、内部由来が分かる名前の docs 追記。
- 既存 training loop、loss、model architecture の大規模変更。
- user / 他エージェント由来の未追跡ファイルの勝手な削除や取り込み。
- generated panel image / graph JSON の Git 追加。

## 5. インタラクション方針

- **回答スタイル:** 日本語、簡潔、見出し＋箇条書き中心。実行結果は command、exit status、主要ログを添える。
- **回答手順:** 前提確認 → 変更範囲 → 実行コマンド → 検証結果 → 残課題の順で報告する。
- **禁止事項・注意:** 未実行の full training を完了済みと書かない。推測を事実として断定しない。dataset/checkpoint の Git 混入を放置しない。
- **秘匿情報の扱い:** GitHub token、ローカル認証情報、外部サービス認証情報、private dataset の取得 URL や資格情報は文書化しない。

## 6. 試行タスク（オンボーディング演習）

> 小さな検証タスクを2〜3件記載してください。理解度を確認するために実施します。

1. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_guyot_dataset.py -q` を実行し、parser / dataset / adapter contract が通ることを確認する。
2. `configs/tree_2D_guyot_train_smoke.yaml` を読み、`DATA.TRAIN_LIMIT: 1`、`DATA.VAL_LIMIT: 1`、repo 外 `TRAIN.SAVE_PATH` が設定されていることを説明する。
3. `PYTHONPATH=. "$TREEFORMER_PYTHON" train_hydra.py --cfg job` を実行し、Hydra config が合成できることを確認する。
4. `git status --ignored --short` で checkpoint / dataset / cache が repo に混入していないことを確認し、残る ignored artifact が CUDA build artifact だけかを報告する。
5. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_graph_augmentations.py -q` を実行し、augmentation contract が通ることを確認する。
6. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_infer_panel_treeformer.py -q` を実行し、checkpoint weight selection と panel rendering helper が通ることを確認する。
7. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_aux_training.py tests/test_hydra_config.py -q` を実行し、aux supervised loss と Hydra config contract が通ることを確認する。
8. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_infer_aux_panel_treeformer.py -q` を実行し、aux panel helper が通ることを確認する。
9. `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_joint_training.py tests/test_tune_graph_optuna.py -q` を実行し、joint graph+aux loop と Optuna report helper が通ることを確認する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 変更した理由、対象ファイル、検証コマンド、結果、残課題を同じ更新に含める。
- **TBDの扱い:** TBD は owner、確認方法、期限または次アクションを併記する。単独の TBD を残さない。
- **レビュー/承認フロー:** 実装 → 最小テスト → artifact 混入確認 → workdoc / docs 更新 → commit。長時間学習は開始前に方針確認する。
- **その他の運用ルール:** `git add -A` は避け、混在 worktree では明示ファイルだけ stage する。生成物 cleanup 後に `git status` を確認する。

## 8. Pretrained Weights

フォーク元 `huntorochi/TreeFormer` の README は pretrained checkpoint を Google Drive folder として公開している。

- source repository: `https://github.com/huntorochi/TreeFormer`
- Google Drive folder: `https://drive.google.com/drive/folders/1QFIwOAESSAF8Uc4it0-cAzBiMMszNJg2?usp=sharing`
- local asset path: `${TREEFORMER_ASSETS_ROOT}/pretrained_weights/fork_source_main/`
- repo には checkpoint を置かない。

取得コマンド:

```bash
export TREEFORMER_ASSETS_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}
mkdir -p "$TREEFORMER_ASSETS_ROOT/pretrained_weights/fork_source_main"

uvx --from gdown gdown --folder \
  'https://drive.google.com/drive/folders/1QFIwOAESSAF8Uc4it0-cAzBiMMszNJg2?usp=sharing' \
  -O "$TREEFORMER_ASSETS_ROOT/pretrained_weights/fork_source_main"
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

TreeFormer の legacy dataloader は split ごとの dataset root を `LoadCNNDataset(parent_path=...)` に渡す。入力は `img/` 配下の RGB 画像で、同名 sample id の `.pt` graph annotation を `data/` から読む。

split root の基本構造:

```text
<split_root>/
├── data/
│   └── <sample_id>.pt
├── img/
│   └── <sample_id>.png
├── seg/
│   └── <sample_id>.png
├── check/
│   └── <sample_id>.png
└── unet/  # legacy fallback only
    └── <sample_id>.png
```

`LoadCNNDataset` が直接読む最小要件:

- `data/<sample_id>.pt`
  - `list_DETR_points_left_up`: normalized 2D node coordinates
  - `DETR_node_collections`: graph connectivity / edge path information
- `img/<sample_id>.png`
  - RGB image input。alpha channel がある場合は先頭 3 channel のみ使う。
- `seg/<sample_id>.png`
  - TPE binary segmentation mask for `train=seg_supervised` / `train=aux_supervised`。背景 + 単一 foreground class の target として扱う。`0/255` PNG は loader が binary float target に変換し、loss 側も `[0, 1]` 外の target を拒否する。

`check/` は補助可視化用、`unet/` は古い dataset 互換の fallback として扱う。新規 private dataset の場合も docs に実パスや生成元を残さず、上記の抽象 layout と file contract だけを書く。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:**
  - repo: current checkout root
  - assets: `${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}`
  - uv venv / Python: `${TREEFORMER_PYTHON:-.venv/bin/python}`
  - raw Guyot extracted dataset: `${TREEFORMER_GUYOT_DATA}` or `${TREEFORMER_ASSETS_ROOT}/datasets/3D2cut_Single_Guyot_extracted`
  - pretrained weights: `${TREEFORMER_ASSETS_ROOT}/pretrained_weights/fork_source_main`
  - smoke outputs: `${TREEFORMER_ASSETS_ROOT}/trained_weights_smoke`

- **実行環境（venv 一本化後の正本）:**
  - 正本は repo 内 `.venv` (Python 3.10)。`just setup-venv` で `uv venv --python 3.10` → `uv sync --inexact --all-groups` → `models/ops` の CUDA ops ビルド → `MultiScaleDeformableAttention` の import / forward check、まで一括で行う。
  - repo 外 `../venv/TreeFormer` (Python 3.10) は**移行前の旧環境**として残置している（削除しない。venv 一本化のロールバック経路）。新規セットアップでは使わない。
  - `justfile:3` の `python` 変数は `env_var_or_default("TREEFORMER_PYTHON", ".venv/bin/python")`。`TREEFORMER_PYTHON` を明示的に unset すれば `just test` / `just lint` 等すべての recipe が repo 内 `.venv` を使う。

- **venv 一本化で実測した罠 5 件（このセッションで実測。書き残さないと再発する）:**
  1. **素の `uv sync` は `MultiScaleDeformableAttention` を削除する。** 必ず `--inexact` を付ける。`uv sync --dry-run --all-groups` は `Would uninstall 1 package - multiscaledeformableattention==1.0` を宣言する（`setup.py install` が `.egg-info` を生成し、uv が「削除可能な distribution」と認識するため）。
  2. **`uv sync` と `uv run` は既定が逆。** `uv sync` の既定は exact（lock にないものを削除する）、`uv run` の既定は inexact（削除しない）。したがって `uv run pytest` は安全だが、**`uv run --exact` を使うと `.so` が消える。**
  3. **`.so` は ABI タグ `cp310` に加え GPU arch `sm_86` にも固定される。** `TORCH_CUDA_ARCH_LIST` 未設定時、torch の `cpp_extension` はビルド時に可視な GPU の compute capability だけを焼く（本環境の RTX A4500 は cc 8.6）。Python バージョンまたは GPU 世代が変わる環境へ移す場合は `models/ops` の再ビルドが必須。
  4. **`import MultiScaleDeformableAttention` を単独で書かない。** `.so` は `libc10.so`（torch の共有ライブラリ）に動的リンクしており、`import torch` を先に実行しないと**健全な環境でも** `ImportError: libc10.so: cannot open shared object file` になる。検証は必ず `import torch, MultiScaleDeformableAttention` の順で行う。
  5. **`uv run pytest` には `PYTHONPATH=.` が要る。** `pyproject.toml` は `[tool.uv] package = false` で project 自体が install されないため、素の `uv run pytest` は失敗する。画面に出るのは `Interrupted: 16 errors during collection` / `16 errors in ...` で、その内訳が `ModuleNotFoundError: No module named 'treeformer_train'` である。`PYTHONPATH=. uv run pytest tests/ -q` なら `99 passed`。`just test` recipe は `PYTHONPATH=.` を明示しているので通る。**この違いを venv 一本化の失敗と誤解しないこと。**

- **`just` の PATH:**
  - 本書と `docs/HYDRA_TRAINING.md` の手順はほぼすべて `just` 前提だが、`just` は `~/.cargo/bin/just` にあり、非ログインシェルでは PATH に載らないことがある。最初に `command -v just` で確認し、無ければ `export PATH="$HOME/.cargo/bin:$PATH"` を通す。

- **代表的なコマンド:**

```bash
export TREEFORMER_PYTHON=${TREEFORMER_PYTHON:-.venv/bin/python}
export TREEFORMER_ASSETS_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$TREEFORMER_PYTHON" -m pytest -p no:cacheprovider tests/test_guyot_dataset.py -q

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$TREEFORMER_PYTHON" -m torch.distributed.run --nproc_per_node=1 --master_port=29532 train_mst.py --config configs/tree_2D_guyot_train_smoke.yaml --cuda_visible_device 0 --local_rank 0

git status --short --branch --untracked-files=all
git status --ignored --short
```

- **依存ライブラリ:** `pyproject.toml` を参照。主要依存は PyTorch 2.6 系、torchvision 0.21 系、scikit-image 0.25 系、NetworkX、Pillow、PyYAML、Hydra、TensorBoard、Schedule-Free。MMCV / OpenMMLab 系は使用しない。
- **連絡先/責任者:** 未定。

> ※この文書は現在の smoke training 完了スコープに基づく。full training の実行結果を得た場合は、config、checkpoint path、主要ログ、失敗時の recovery 手順を追記する。
