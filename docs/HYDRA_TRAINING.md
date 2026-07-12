# Hydra Training Infrastructure

This repository now includes a Hydra-managed training entrypoint for TreeFormer experiments.

## Main entrypoint

```bash
export TREEFORMER_PYTHON=${TREEFORMER_PYTHON:-.venv/bin/python}
export TREEFORMER_ASSETS_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}

PYTHONPATH=. "$TREEFORMER_PYTHON" train_hydra.py --cfg job
PYTHONPATH=. "$TREEFORMER_PYTHON" train_hydra.py \
  optimizer=muon_schedulefree ema=default train=dry_run \
  runtime.device=cpu runtime.fail_if_cuda_unavailable=false \
  logging=disabled checkpoint.enabled=false
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$TREEFORMER_PYTHON" train_hydra.py \
  optimizer=muon_schedulefree ema=default \
  TRAIN.EPOCHS=1 DATA.TRAIN_LIMIT=1 DATA.VAL_LIMIT=1 DATA.BATCH_SIZE=1
```

The new entrypoint converts Hydra config sections back to the legacy `DATA`, `MODEL`, `TRAIN`, and `log` object contract used by the existing TreeFormer model, loss, dataset, and epoch code. Existing YAML-based scripts are retained.

The default smoke config uses a repo-external synthetic Guyot-format fixture under `${TREEFORMER_ASSETS_ROOT}/datasets/`. Use `DATA.DATA_PATH=...` or `TREEFORMER_PRIVATE_DATA=...` to run against another dataset. Keep private dataset paths out of committed docs and work records. Local `.env` files may be used by an operator's shell, but they are ignored by Git and must not be committed.

## Pretrained private-data smoke template

For a legacy TreeFormer-format RGB + 2D graph dataset, pass the dataset root through an environment variable and keep the concrete path out of committed files:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>

just cfg-private-pretrained-gpu-batch12
just smoke-private-pretrained-gpu-batch12
```

The prepared smoke template uses:

- GPU training: `runtime.device=cuda`
- batch size: `DATA.BATCH_SIZE=12`
- smoke limits: `DATA.TRAIN_LIMIT=24`, `DATA.VAL_LIMIT=12`
- image/graph size cap: `DATA.MAX_SIZE=128`
- optimizer: `optimizer=muon_schedulefree`
- EMA: `ema=default`, with `ema.device=null`, so EMA shadow weights stay on GPU
- TensorBoard: `logging=tensorboard`
- pretrained checkpoint: fork-source `grapevein/checkpoint_ours.pkl`, loaded with `checkpoint.pretrained_key=net`

`cfg-private-pretrained-gpu-batch12` only composes and prints the Hydra config. It does not initialize the model or start training.

With `ema=default`, EMA shadow weights and EMA validation stay on GPU. On RTX A4500, batch size 12 with `DATA.MAX_SIZE=128` used about 3.1GiB for the measured train step; the practical estimate with GPU EMA is about 3.5-4.0GiB. `nvidia-smi` reports total GPU memory across all processes, so separate concurrent jobs must be excluded before treating it as TreeFormer-only VRAM.

## Aux Supervised Map Training

When graph annotations are too sparse or too rough to make `val/smd` meaningful, use `train=aux_supervised` first. This mode keeps the RGB input path, adds a lightweight dense prediction head on the encoder feature map, and disables graph decoder output for the training step.

The aux head predicts four dense channels:

- one binary foreground segmentation logit; background is represented by target `0`
- node heatmap logit
- PAF x direction
- PAF y direction

The training loss is:

- `train/aux_seg_bce`: binary BCE-with-logits against the dataloader external `seg/` mask
- `train/aux_heatmap_mse`: MSE between sigmoid heatmap output and generated node heatmap
- `train/aux_paf_l1`: masked L1 between tanh PAF output and generated PAF vectors

Validation uses the same direct supervision and checkpoints on `val/aux_total_loss`. Graph losses are not computed, and the graph SMD validator is skipped. This makes the first question concrete: can the network learn the mask / heatmap / direction fields from RGB before asking it to output a clean graph.

For the current stabilization work, prefer independent dense modes before the full aux-map objective:

- `train=seg_only`: segmentation loss only. It treats segmentation as background + one foreground target class, sets detail / heatmap / PAF loss weights to zero, and checkpoints on `val/seg_soft_dice_score`.
- `train=seg_heatmap`: segmentation plus node heatmap. It keeps graph output and graph losses disabled, adds `W_AUX_HEATMAP=1`, keeps PAF loss at zero, and uses `DATA.AUX_TARGET_MODE=seg_heatmap` so `FastSegSupervisedDataset` generates node heatmaps from the split graph annotation.
- `train=seg_heatmap_paf`: segmentation plus node heatmap plus PAF / edge-direction supervision. It keeps graph output and graph losses disabled, uses `DATA.AUX_TARGET_MODE=seg_heatmap_paf`, trains the two PAF channels with masked L1, and checkpoints on `val/aux_total_loss`.
- `train=seg_supervised`: legacy compatibility mode from the earlier stabilization work. It is segmentation plus weak external-mask-derived detail boundary regularization.

The segmentation target must come from the split-local external TPE binary mask directory, `seg/` preferred and `unet/` accepted only as a legacy fallback. The graph-derived raster mask is not used as the segmentation target in these modes.

Config-only check:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just cfg-private-seg-only
just cfg-private-seg-heatmap
just cfg-private-seg-heatmap-paf
just cfg-private-seg-supervised
just cfg-private-aux-supervised
```

Short GPU smoke:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just smoke-private-seg-only
just smoke-private-seg-heatmap
just smoke-private-seg-heatmap-paf
just smoke-private-seg-supervised
just smoke-private-aux-supervised
```

Full no-geometry aux stage:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just cache-private-fast-seg
just train-private-seg-only
just train-private-seg-heatmap
just train-private-seg-heatmap-paf
just train-private-seg-supervised
just train-private-aux-supervised
```

After an aux stage, render validation panels before deciding the next step:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_AUX_CHECKPOINT=<aux_stage_best_checkpoint>
export TREEFORMER_AUX_PANEL_OUTPUT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/aux_inference_panels/<stage_name>

just infer-aux-panels
```

`infer_aux_panel_treeformer.py` writes one `<sample_id>_aux_panel.png` per image. The panels include input and GT/pred segmentation overlays. The GT segmentation overlay should be read from the same external TPE mask contract as training. When a checkpoint has a fifth aux channel, panels also show `GT detail boundary target` and `Pred detail boundary head`; otherwise derived boundary maps stay hidden unless `--show-derived-detail` is passed explicitly. Heatmap and PAF panels are hidden automatically when their checkpoint loss weights are zero; pass `--show-untrained-maps` only for debugging raw aux channels.

## Virtual-Root Forest Graph Mode

`docs/TreeFormer_virtual_root_forest_formalization.pdf` and `.tex` define the current virtual-root forest contract. The short version is that real nodes are not forced into one MST. A dummy root is added, maximum spanning tree is solved on the augmented graph, and root edges are removed to produce a forest on real nodes.

Use `train=virtual_root` only with graph annotations that provide virtual-root metadata:

- `component_id`: `LongTensor[N]`
- `component_count`: integer or scalar tensor
- `root_node_indices`: optional representative real node indices
- `root_edge_index`: optional augmented root edge index
- `graph_topology`: `virtual_root_forest_v1`

The legacy reader and fast segmentation reader can preserve this metadata when `DATA.FOREST_METADATA=true`. In `train=virtual_root`, `DATA.STRICT_VIRTUAL_ROOT_METADATA=true` is set, so missing `component_id` is a data error. It must not silently fall back to single-tree MST.

The virtual-root model path adds `MODEL.ROOT_HEAD.ENABLED=true`, which emits `pred_root_logits` with shape `[B, OBJ_TOKEN]`. Pretrained checkpoints that do not contain this head require `checkpoint.pretrained_strict=false`.

For qualitative graph panels, `infer_panel_treeformer.py` accepts `MODE=vr-mst`. JSON output includes optional `postprocessor_mode`, `root_edges_node_indices`, `component_id`, and `augmented_edges` fields for audit.

Operational notes:

- Use `augmentation=disabled` first. Do not use geometric, deformation, crop, rotate, affine, perspective, or elastic DA for this stage.
- `checkpoint.pretrained_strict=false` is expected because the pretrained graph model has no aux head parameters.
- `DATA.SEGMENTATION_TARGET_SOURCE=external_mask` is expected for `train=seg_supervised` and `train=aux_supervised`. Missing `seg/<sample_id>.png` or legacy `unet/<sample_id>.png` is a configuration error, not a reason to fall back silently to graph-derived labels.
- External mask images may be stored as `0/255` PNGs; the loader thresholds them to binary float targets before BCE / Dice / Focal loss, and the loss code rejects targets outside `[0, 1]`.
- The default segmentation loss uses binary BCE + Dice with Focal disabled. `AUX_SEG_POS_WEIGHT=auto` is capped conservatively so rare foreground does not cause broad positive overprediction.
- `train=seg_only`, `train=seg_heatmap`, and `train=seg_heatmap_paf` use `FastSegSupervisedDataset` by default. `DATA.AUX_TARGET_MODE=seg_only` returns zero-valued unused PAF/heatmap tensors. `DATA.AUX_TARGET_MODE=seg_heatmap` generates only node heatmap targets and leaves PAF tensors zero. `DATA.AUX_TARGET_MODE=seg_heatmap_paf` generates both node heatmap and direction targets. For precomputed repo-external cache, use `just cache-private-fast-seg`, `just cache-private-fast-seg-heatmap`, or `just cache-private-fast-seg-heatmap-paf` respectively.
- Heatmap and direction supervision is restricted to the external segmentation foreground. `AUX_HEATMAP_MASK_OUTSIDE_WEIGHT=0` means heatmap loss ignores mask-outside pixels. `AUX_DIRECTION_TARGET_SOURCE=mask_skeleton` skeletonizes the external mask, estimates local tangents, and propagates them to the foreground using an EDT nearest-skeleton index map. Endpoint/junction neighborhoods are excluded because one direction cannot represent multiple branches. The default `AUX_DIRECTION_ENCODING=double_angle` stores `[cos(2*theta), sin(2*theta)]`, so opposite tangent signs are equivalent. The old sparse graph-edge PAF remains available only with explicit `graph_edges` selection.
- Heatmap supervision is ablation-ready. The shipped `seg_heatmap` / `seg_heatmap_paf` / `joint_virtual_root_aux` configs keep the Gaussian MSE baseline (`W_AUX_HEATMAP_MSE=1`, focal/coord/peakness disabled) so comparisons remain interpretable. The opt-in `+ablation=heatmap_peak_focused` recipe enables target-local-peak focal positives, ridge suppression, local soft-argmax coordinate loss, variance suppression, and a peakness margin loss with conservative weights. This separates the documented proposal from the baseline until it demonstrates improved node recall/precision without reducing segmentation quality.
- The recently added PAF angular direction loss is enabled at a modest weight in `seg_heatmap_paf` and `joint_virtual_root_aux`: `W_AUX_PAF_L1=1.0` keeps masked L1 as the primary PAF term, and `W_AUX_PAF_ANGULAR=0.25` adds a cosine-similarity direction loss on top.
- New objective peak-quality validation metrics: `val/heatmap_node_recall`, `val/heatmap_node_precision`, `val/heatmap_duplicate_peak_rate`, and `val/heatmap_background_peaks_per_image`, evaluated with `AUX_HEATMAP_EVAL_PEAK_THRESHOLD` and `AUX_HEATMAP_EVAL_MATCH_RADIUS`. These directly measure whether the heatmap produces one clean peak per node rather than a ridge or multiple duplicate peaks. Predictions and target peaks are restricted to the configured segmentation foreground.
- `train=seg_supervised` adds a weak optional STDC-style detail boundary auxiliary head as the fifth aux channel. Its target is a multi-scale Laplacian boundary map derived from the external segmentation mask, not a graph edge or PAF. `W_AUX_DETAIL=0.1` keeps it as a light boundary-sharpening regularizer while BCE + Dice segmentation remains the primary objective.
- For full segmentation-only training, generate repo-external cache first with `just cache-private-fast-seg`, then run `just train-private-seg-supervised`. Cache format v3 includes the direction-target source/encoding and omits the unused detail-boundary payload, so regenerate older caches. Keep the worker count an explicit experiment variable: the measured local disk-cache joint workload is fastest with `DATA.NUM_WORKERS=0`, while uncached target generation benefits from workers. Set `TREEFORMER_SEG_CACHE_MODE=none` to bypass disk cache.
- `FastSegSupervisedDataset` defaults to `DATA.SEG_RESIZE_POLICY=legacy_half`, matching the legacy loader behavior that halves the raw image before applying `DATA.MAX_SIZE`. For a raw 800x600 image, `DATA.MAX_SIZE=128` becomes 128x96 and `DATA.MAX_SIZE=512` still caps at 400x300. Use `DATA.SEG_RESIZE_POLICY=full` with `DATA.MAX_SIZE=512` to train at 512x384, or `DATA.SEG_RESIZE_POLICY=full` with `DATA.MAX_SIZE=800` to keep the original 800x600 shape.
- Aux/seg stages may keep EMA updates enabled, but use `ema.evaluate=false` for validation and best-checkpoint selection because the aux head is newly initialized and `decay=0.9999` changes too slowly during short stages.
- To initialize a dense stage from a previous Hydra `best.pt`, set `TREEFORMER_PRETRAINED_CHECKPOINT=<best.pt>` and `TREEFORMER_PRETRAINED_KEY=model`. Fork-source checkpoints keep the default `TREEFORMER_PRETRAINED_KEY=net`.
- RGB input is normalized by the legacy loader with `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`, so model input is scaled from `[0, 1]` to `[-1, 1]`.
- In `train=seg_only` and `train=seg_supervised`, `W_AUX_HEATMAP=0` and `W_AUX_PAF=0`; heatmap and PAF channels are not trained in those stages. In `train=seg_heatmap`, `W_AUX_HEATMAP=1` and `W_AUX_PAF=0`. In `train=seg_heatmap_paf`, `W_AUX_HEATMAP=1` and `W_AUX_PAF=0.25`, so the edge-direction channels are trained together with segmentation and node heatmaps. The aux output layout is segmentation, heatmap, PAF-x, PAF-y, optional detail boundary. `train=aux_supervised` keeps the original 4-channel layout and has `W_AUX_DETAIL=0`.
- `DATA.MAX_SIZE=128` with `legacy_half` is a smoke-only contract. For a thin-structure full run, explicitly use the documented full-resolution contract (`DATA.MAX_SIZE=640`, `DATA.SEG_RESIZE_POLICY=full`) and select batch size from a one-batch GPU preflight; do not silently reuse the 128px smoke override.
- Monitor `val/seg_soft_dice_score`, `val/seg_dice_score`, `val/seg_iou`, `val/seg_precision`, `val/seg_recall`, `val/pred_positive_rate`, and when heatmap is enabled `val/masked_heatmap_mae`; do not interpret `val/smd` for this mode because graph validation is intentionally skipped. `val/seg_dice_score` and IoU are hard-threshold metrics at `AUX_SEG_THRESHOLD`; early runs may show zero while soft Dice and loss still improve.
- For heatmap/direction ablations, also monitor `val/heatmap_peak_contrast`, `val/heatmap_peak_mean`, `val/heatmap_nonpeak_foreground_mean`, and `val/direction_angular_error_deg`. The desired direction is higher peak contrast and lower angular error while preserving segmentation IoU/Dice. Direction metrics are computed only on the valid mask-skeleton foreground.

The optional graph connection is controlled by `MODEL.AUX_HEAD.GRAPH_CONDITIONING`. `none` preserves the independent aux and graph paths. `aux_feature` injects the lightweight aux trunk feature into the first graph feature level through a learned residual projection; it does not concatenate full-resolution logits. `train=joint_virtual_root_aux` enables this connection for the graph stage.

## Dense Aux Ablation Study

Use this study when segmentation and PAF look learnable but node heatmap remains ridge-like instead of point-like. The study intentionally avoids architecture changes. It varies only target sigma, heatmap loss composition, ridge suppression, and segmentation task weight.

Initial ablation matrix:

| ID | Hydra override | Target sigma | Heatmap loss | Seg weight | Question |
|---|---|---:|---|---:|---|
| A0 | `+ablation=heatmap_mse_baseline` | 3.0 | MSE | 1.0 | Reproduce the current 640x480 seg+heatmap+PAF behavior. |
| A1 | `+ablation=heatmap_sigma1_5_mse` | 1.5 | MSE | 1.0 | Does a smaller Gaussian target make heatmap peaks less ridge-like by itself? |
| A2 | `+ablation=heatmap_focal` | 1.5 | `0.25*MSE + focal` | 1.0 | Does CenterNet-style focal loss improve point contrast without hurting segmentation? |
| A3 | `+ablation=heatmap_focal_ridge` | 1.5 | `0.25*MSE + focal + 0.1*ridge` | 1.0 | Does foreground non-peak suppression reduce continuous heatmap bands? |
| A4 | `+ablation=heatmap_focal_ridge_seg_low` | 1.5 | `0.25*MSE + focal + 0.1*ridge` | 0.5 | Does reducing segmentation loss pressure help heatmap pointness? |
| A5 | `+ablation=heatmap_peak_focused` | 3.0 | `0.25*MSE + focal + coord + variance + peakness + 0.05*ridge` | 1.0 | Does the formalized local point objective improve node peak quality without sacrificing segmentation? |

For sigma-changing ablations, regenerate or extend the repo-external cache with the same sigma used by the training config:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_MAX_SIZE=640
export TREEFORMER_SEG_RESIZE_POLICY=full
export TREEFORMER_BATCH_SIZE=8

# A0
export TREEFORMER_ABLATION=heatmap_mse_baseline
export TREEFORMER_HEATMAP_SIGMA=3.0
just cache-private-fast-seg-heatmap-paf-ablation
just train-private-seg-heatmap-paf-ablation

# A1-A4
export TREEFORMER_HEATMAP_SIGMA=1.5
for name in heatmap_sigma1_5_mse heatmap_focal heatmap_focal_ridge heatmap_focal_ridge_seg_low; do
  export TREEFORMER_ABLATION="$name"
  just cache-private-fast-seg-heatmap-paf-ablation
  just train-private-seg-heatmap-paf-ablation
done
```

Use `smoke-private-seg-heatmap-paf-ablation` before full runs when changing the matrix. After runs finish:

```bash
just summarize-private-aux-ablation
export TREEFORMER_AUX_CHECKPOINT=<ablation_best_checkpoint>
export TREEFORMER_AUX_PANEL_OUTPUT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/aux_inference_panels/<ablation_name>
just infer-aux-panels
```

Decision rule:

- Prefer the run with the best combination of high `val/heatmap_peak_contrast`, low `val/masked_heatmap_mae`, stable or improved `val/seg_iou` / `val/seg_dice_score`, and non-regressed `val/paf_masked_l1`.
- Do not choose a run solely because `val/aux_total_loss` is lower; focal/ridge loss changes the numeric scale.
- If all loss-only heatmap ablations remain ridge-like, move to the node-center/offset/ranking profile below before trying query-level supervision. Do not add more coefficients to local soft-argmax, variance, or peakness terms.

## Joint Virtual-Root Graph + Aux Tuning

`train=joint_virtual_root_aux` is the graph-stage mode for final reconstruction tuning. It keeps graph output enabled and trains the virtual-root graph objective together with every currently available dense auxiliary head:

- virtual-root forest metadata and `POSTPROCESSOR_MODE=vr-mst`
- root head and root loss
- graph losses: boxes, class, cards, nodes, virtual-root edges, root
- dense aux losses: external-mask segmentation, STDC-style detail boundary, node heatmap, PAF / edge-direction
- segmentation-aware heatmap and PAF masking

The joint objective is:

```text
joint_total = graph_total + TRAIN.W_JOINT_AUX * aux_total
```

This mode is intentionally separate from `train=seg_only`, `train=seg_heatmap`, and `train=seg_heatmap_paf`. Dense-only modes keep graph loss at zero to verify target quality. `train=joint_virtual_root_aux` is used after dense targets are known to be learnable and the goal is graph reconstruction.

`tune_graph_optuna.py` explores heatmap sigma 3.0 and 1.5. With `DATA.SEG_CACHE_MODE=disk`, create the matching cache roots first with `just cache-private-joint-virtual-root-aux-optuna`; the runner selects `heatmap_sigma_3_0/` or `heatmap_sigma_1_5/` explicitly for each trial. A missing matching cache is an error, never a fallback to another sigma.

### Validation interval

`TRAIN.VAL_INTERVAL` controls how often the validation pass runs. The default is `1`, which preserves validation every epoch. For longer training runs, set `TRAIN.VAL_INTERVAL=5` or `TRAIN.VAL_INTERVAL=10`; validation runs on those epoch boundaries and always on the final epoch. Training loss, learning rate, epoch time, and `validation/ran` are logged every epoch. On skipped epochs, checkpoint updates are deferred until the next validation epoch because the checkpoint manager requires the configured validation metric.

The same setting can be supplied through the standard training recipes:

```bash
TREEFORMER_VAL_INTERVAL=5 just train-private-joint-virtual-root-aux
```

An explicit Hydra override takes precedence when invoking `train_hydra.py` directly:

```bash
PYTHONPATH=. .venv/bin/python train_hydra.py \
  train=joint_virtual_root_aux TRAIN.VAL_INTERVAL=10
```

### Optional GPU SMD validation backend

The default `TRAIN.SMD_BACKEND=legacy` preserves the original NetworkX/CPU
metric. An optional `geomloss_gpu` backend keeps the graph post-processing
unchanged, samples each graph into a fixed-size point cloud with PyTorch, and
computes the regularized transport cost with GeomLoss on the same device as the
model output. It is opt-in because the point-cloud sampling convention and
GeomLoss regularization are not guaranteed to be numerically identical to the
legacy implementation.

Install the optional dependency into the repo-local uv environment:

```bash
uv pip install --python .venv/bin/python 'geomloss>=0.2.6,<0.3'
```

Use it for a comparison run with explicit overrides:

```bash
TREEFORMER_SMD_BACKEND=geomloss_gpu \
TREEFORMER_SMD_N_POINTS=500 \
TREEFORMER_SMD_BLUR=0.01 \
just train-private-joint-virtual-root-aux
```

Before using this backend for checkpoint selection, compare legacy and GPU
validation on the same checkpoint. Record finite outputs, metric ranking, best
epoch agreement, and representative panels. Do not report GeomLoss scores as
bitwise-equivalent legacy SMD scores.

Recommended smoke:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_MAX_SIZE=640
export TREEFORMER_BATCH_SIZE=2
export TREEFORMER_SEG_RESIZE_POLICY=full

just cfg-private-joint-virtual-root-aux
just smoke-private-joint-virtual-root-aux
```

Recommended full single run:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_MAX_SIZE=640
export TREEFORMER_BATCH_SIZE=8
export TREEFORMER_EPOCHS=100
export TREEFORMER_SEG_CACHE_MODE=disk
export TREEFORMER_SEG_CACHE_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/cache/fast_seg/<cache_name>

just cache-private-fast-seg-heatmap-paf
just train-private-joint-virtual-root-aux
```

For storage control, joint/Optuna recipes set `checkpoint.save_last=false`; keep `best.pt`, TensorBoard scalar logs, and report files. If resume-from-last is required for a specific run, override `checkpoint.save_last=true` explicitly and clean it up after the run.

### Optuna

Optuna is optional. Install it into the uv environment only when tuning:

```bash
just install-optuna
```

Default tuning command:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_MAX_SIZE=640
export TREEFORMER_BATCH_SIZE=8
export TREEFORMER_EPOCHS=100
export TREEFORMER_OPTUNA_TRIALS=20
export TREEFORMER_SEG_CACHE_MODE=disk
export TREEFORMER_SEG_CACHE_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/cache/fast_seg/<cache_name>

just tune-private-joint-virtual-root-aux
```

For a short runner smoke, keep the same command shape and constrain the dataset:

```bash
export TREEFORMER_OPTUNA_TRIALS=1
export TREEFORMER_EPOCHS=1
export TREEFORMER_BATCH_SIZE=1
export TREEFORMER_TRAIN_LIMIT=2
export TREEFORMER_VAL_LIMIT=1
export TREEFORMER_SEG_CACHE_MODE=none

just tune-private-joint-virtual-root-aux
```

The runner writes repo-external outputs under `${TREEFORMER_OPTUNA_OUTPUT:-${TREEFORMER_ASSETS_ROOT}/optuna/joint_virtual_root_aux}`:

- `optuna_study.db`: resumable SQLite study
- `trials.csv`: machine-readable trial table for other agents
- `report.md`: Japanese summary for other agents
- `best_trial_overrides.yaml`: winning parameter values for manual review
- `training/runs/<trial>_<seed>/checkpoints/best.pt`: best checkpoint for each completed trial

The objective is maximized:

```text
-val/smd + 0.10*val/seg_iou + 0.05*val/heatmap_peak_contrast - 0.05*val/paf_masked_l1
```

This is a pragmatic proxy until the heavier graph evaluator is wired into the trial loop. `val/smd` remains the dominant term, while dense aux metrics prevent Optuna from selecting graph-only improvements that destroy segmentation, node heatmap, or edge-direction supervision.

The search space is intentionally conservative:

- `TRAIN.LR`, `TRAIN.LR_BACKBONE`
- `TRAIN.W_NODE`, `TRAIN.W_EDGE`, `TRAIN.W_ROOT`
- `TRAIN.W_JOINT_AUX`
- `TRAIN.W_AUX_SEG`, `TRAIN.W_AUX_DETAIL`, `TRAIN.W_AUX_HEATMAP`, `TRAIN.W_AUX_PAF`
- heatmap profile: MSE baseline, sigma 1.5 MSE, focal, focal + ridge
- `TRAIN.CLIP_MAX_NORM`

Do not commit Optuna DBs, trial checkpoints, TensorBoard events, CSV reports, or generated panels. The Markdown report must not include private dataset paths; pass dataset locations through environment variables only.

## Augmentation

The current private-data curriculum intentionally avoids geometric and deformation augmentation. Use `augmentation=disabled` for the stabilization stage and `augmentation=photometric_opencv` only after the no-DA baseline behaves well.

`augmentation=regularized` and `augmentation=geometry_mild` remain available as implementation experiments, but they are not part of the current default curriculum.

The augmentation code is split by contract:

- photometric transforms are image-only and leave graph nodes / edges unchanged.
- affine and elastic transforms update the RGB image and normalized node coordinates from the same transform field.
- edge topology is preserved; transforms that would move any node outside the image are rejected for that sample.

`augmentation=regularized` also sets `DATA.LEGACY_ROTATE=false`, so geometry regularization is owned by the graph-aware augmentation module instead of stacking on top of the legacy always-rotate path.

Photometric backend selection:

- `augmentation=regularized` uses `backend=opencv`, so the default regularized DA path avoids AGPL dependency exposure.
- `augmentation=regularized_albumentationsx` explicitly opts into the AlbumentationsX-compatible `albumentations` module.
- `allow_fallback=true` falls back to the OpenCV implementation when AlbumentationsX is unavailable.
- `augmentation=photometric_opencv` runs image-only optical DA.
- `augmentation=geometry_mild` remains an implementation experiment; it is not part of the current curriculum.

AlbumentationsX is optional because it has a separate dual-license model and can require platform-specific binary dependencies. To enable that backend in the active venv through the project dependency group:

```bash
uv pip install --python "$TREEFORMER_PYTHON" --project . --group albumentationsx
# or
just install-albumentationsx
```

For a short GPU smoke run with pretrained weights and photometric-only DA:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "$TREEFORMER_PYTHON" train_hydra.py \
  augmentation=photometric_opencv optimizer=muon_schedulefree ema=default \
  logging=tensorboard checkpoint.pretrained="$TREEFORMER_PRETRAINED_CHECKPOINT" \
  checkpoint.pretrained_key=net checkpoint.pretrained_strict=true \
  DATA.DATASET=treeformer-2D DATA.DATA_PATH="$TREEFORMER_PRIVATE_DATA" \
  DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 \
  DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.EPOCHS=3
```

The smoke recipe runs 3 epochs with batch size 12, GPU EMA, TensorBoard, Muon + ScheduleFree, and repo-external output paths. Keep the dataset root in the environment variable and out of committed docs.

## Training Curriculum

The first curriculum should be stage-based, using checkpoint resume between stages. This keeps the legacy training loop simple and makes each stage auditable in TensorBoard and checkpoint folders.

Recommended initial curriculum for a pretrained private legacy TreeFormer-format dataset:

| Stage | Config | Epochs | LR / backbone LR | Purpose | Stop rule |
|---|---:|---:|---:|---|---|
| S0. Segmentation | `train=seg_only augmentation=disabled` | 20 | `1e-4` / `3e-5` | Confirm the RGB encoder can learn the external binary target mask before adding heatmap / PAF / graph objectives. | Continue only if `val/seg_soft_dice_score` or `val/seg_total_loss` improves, then check hard-threshold Dice/IoU and foreground rate for calibration. |
| H0. Seg + Heatmap | `train=seg_heatmap augmentation=disabled` | 20 | `1e-4` / `3e-5` | Add node heatmap supervision while keeping graph output and PAF disabled. | Continue only if heatmap error declines without collapsing segmentation metrics. |
| P0. Seg + Heatmap + PAF | `train=seg_heatmap_paf augmentation=disabled` | 20 | `1e-4` / `3e-5` | Add edge-direction / PAF supervision after segmentation and heatmap are learnable. | Continue only if `val/aux_total_loss` and `val/paf_masked_l1` decline without collapsing segmentation metrics. |
| A0. Aux maps | `train=aux_supervised augmentation=disabled` | 20 | `1e-4` / `3e-5` | Legacy 4-channel aux objective for comparison. | Continue only if `val/aux_total_loss` declines without collapsing segmentation metrics. |
| G0. Graph stabilize | `augmentation=disabled` | 20 | `3e-5` / `1e-5` | Re-enable graph output only after aux maps show learnable supervision. | Continue only if train loss declines and `val/smd` does not spike. |
| G1. Optical | `augmentation=photometric_opencv` | 80 | `5e-5` / `1.5e-5` | Learn robustness to illumination, noise, blur, and color shifts while graph labels stay unchanged. | Use best checkpoint if `val/smd` improves; otherwise return to previous best. |

Operational notes:

- Use `best.pt` from the previous stage as `checkpoint.resume` for the next stage, and set `checkpoint.pretrained=null` after Stage 0.
- Keep `DATA.BATCH_SIZE=12`, `DATA.MAX_SIZE=128`, GPU EMA state updates, TensorBoard, and Muon + ScheduleFree unless a stage shows instability. For aux/seg stages, keep `ema.evaluate=false` so validation follows live model weights.
- Do not use random crop, rotate, scale, affine, perspective, elastic, or graph-deformation augmentation in this curriculum.
- Prefer the OpenCV photometric config for Stage 1. Use `regularized_albumentationsx` only after explicit license acceptance and optional dependency installation, and only for image-only photometric experiments.
- Do not mix private dataset paths into docs, commit messages, or work records; pass them through `TREEFORMER_PRIVATE_DATA`.

Config-only checks are available:

```bash
just cfg-private-seg-supervised
just cfg-private-aux-supervised
just cfg-private-curriculum-stage0
export TREEFORMER_CURRICULUM_RESUME=<previous_stage_best_or_last_checkpoint>
just cfg-private-curriculum-stage1
```

## Config groups

| Group | Files | Purpose |
|---|---|---|
| `data` | `guyot_smoke`, `guyot_full` | Guyot dataset paths, limits, workers, seed |
| `augmentation` | `disabled`, `photometric_opencv`, `regularized`, `regularized_albumentationsx`, `geometry_mild` | Image-only photometric DA and graph-aware affine / elastic DA |
| `model` | `treeformer_2d` | Existing 2D TreeFormer architecture settings |
| `train` | `default`, `dry_run`, `seg_only`, `seg_heatmap`, `seg_heatmap_paf`, `seg_supervised`, `aux_supervised`, `virtual_root`, `joint_virtual_root_aux` | Epochs, loss weights, LR, save path, graph-vs-aux training mode |
| `optimizer` | `adamw_step`, `schedulefree_adamw`, `muon_schedulefree` | Optimizer and scheduler selection |
| `logging` | `tensorboard`, `disabled` | TensorBoard event writing |
| `ema` | `disabled`, `default` | EMA update/evaluation behavior |
| `checkpoint` | `default` | last/best/periodic checkpoint policy |
| `distributed` | `single`, `ddp` | Single-process or torchrun/DDP execution |
| `ablation` | `heatmap_mse_baseline`, `heatmap_sigma1_5_mse`, `heatmap_focal`, `heatmap_focal_ridge`, `heatmap_focal_ridge_seg_low`, `heatmap_native_stride4`, `heatmap_native_stride4_offset`, `heatmap_center_offset_rank` | Opt-in dense aux ablation overrides, applied as `+ablation=<name>` |

The `data` group lives in `conf/data/` and is required by `conf/config.yaml` defaults. It is tracked in Git; a checkout missing it cannot compose any Hydra config.

## Aux Inference Panels

Use `infer_aux_panel_treeformer.py` for dense aux output inspection. It reads current Hydra checkpoints, including embedded resolved config and EMA shadow weights. With `--weights auto`, EMA weights are preferred.

For direct invocation:

```bash
PYTHONPATH=. "$TREEFORMER_PYTHON" infer_aux_panel_treeformer.py \
  --legacy-split-root "$TREEFORMER_PRIVATE_DATA/val" \
  --checkpoint "$TREEFORMER_AUX_CHECKPOINT" \
  --output-dir "$TREEFORMER_AUX_PANEL_OUTPUT" \
  --device cuda \
  --max-size 128 \
  --limit 10 \
  --save-json
```

The optional JSON summary stores compact scalar statistics only. Generated panels and summaries must stay under `${TREEFORMER_ASSETS_ROOT}` or another repo-external artifact directory. Detail boundary panels are model-output panels only when the checkpoint has a fifth aux channel; use `--show-derived-detail` only for debugging segmentation-derived boundaries. For segmentation-only checkpoints, heatmap/PAF maps are not trained and are omitted by default.

## Fast Segmentation Dataset Cache

`train=seg_supervised` is bottlenecked by target preparation if it uses the legacy TreeFormer loader. The fast segmentation path avoids graph-derived dense target generation and can optionally read preprocessed image/mask tensors from a repo-external cache.

Generate cache:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_SEG_CACHE_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/cache/fast_seg/<cache_name>

just cache-private-fast-seg
```

Direct CLI:

```bash
PYTHONPATH=. "$TREEFORMER_PYTHON" generate_fast_seg_cache.py \
  --dataset-root "$TREEFORMER_PRIVATE_DATA" \
  --cache-root "$TREEFORMER_SEG_CACHE_ROOT" \
  --splits train val \
  --max-size 128 \
  --resize-policy legacy_half
```

For high-resolution 512x384 training from 800x600 RGB images, use:

```bash
export TREEFORMER_MAX_SIZE=512
export TREEFORMER_SEG_RESIZE_POLICY=full
export TREEFORMER_SEG_CACHE_ROOT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/cache/fast_seg/<cache_name>_max512_full

just cache-private-fast-seg
just train-private-seg-supervised
```

Representative timing with `DATA.MAX_SIZE=128` and `BATCH_SIZE=12`:

| Loader path | Train DataLoader pass |
|---|---:|
| legacy `LoadCNNDataset`, workers=0 | 7.6-8.0s |
| fast seg, no disk cache, workers=0 | 3.4s |
| fast seg, disk cache, workers=0 | 0.21s |
| fast seg, disk cache, workers=4 | 0.43-0.61s |

End-to-end one-epoch GPU timing with fast disk cache and 4 workers:

- train epoch with EMA: about 2.0s
- validation epoch: about 0.4s
- checkpoint save still costs about 0.4-0.8s depending on whether both `best.pt` and `last.pt` are written

Keep cache files outside Git. They are derived artifacts and may contain dataset-derived tensors.

### Native Heatmap Grid

The standard heatmap target is image-resolution (`DATA.AUX_HEATMAP_TARGET_STRIDE=1`). For point-like node supervision at 640x480, use the opt-in `+ablation=heatmap_native_stride4` profile. It changes only the heatmap contract:

- the dense aux decoder reconstructs a stride-4 feature grid from TreeFormer's first stride-8 feature;
- the node heatmap is projected directly from that decoder feature with a `1x1` convolution;
- segmentation remains a dense decoder output and edge direction retains its own output tower;
- the heatmap target is generated at stride 4 with `AUX_HEATMAP_SIGMA=1.0` measured in native-grid cells;
- loss and peak metrics consume `aux_heatmap_native` directly. They do not supervise a full-resolution interpolation.

Generate a separate cache before selecting the profile because cache format v4 includes the target stride:

```bash
export TREEFORMER_MAX_SIZE=640
just cache-private-native-heatmap-stride4
```

Then compose or train with `train=seg_heatmap_paf +ablation=heatmap_native_stride4`. The panel renderer upsamples the native target and prediction only for display. The 9/10-tuple legacy collate contract remains unchanged.

This first comparison intentionally has no NMS or offset loss. If its panel still produces ridge-like peaks, use `+ablation=heatmap_native_stride4_offset`. It adds a direct decoder `2x1` offset projection, learns nearest-cell sub-pixel offsets from existing node coordinates, and applies 3x3 NMS plus those offsets only during inference. NMS candidates are restricted to predicted segmentation foreground. The heatmap head itself remains a direct decoder `1x1` projection.

The offset profile reuses the same stride-4 cache because offsets are derived from the existing per-sample node coordinates at loss time:

```bash
export TREEFORMER_ABLATION=heatmap_native_stride4_offset
export TREEFORMER_SEG_CACHE_ROOT=${TREEFORMER_NATIVE_HEATMAP_CACHE_ROOT:-${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/cache/fast_seg/native_heatmap_stride4_v4}
just train-private-seg-heatmap-paf-ablation
```

### Node-center + offset + structured-ranking diagnostic

`+ablation=heatmap_center_offset_rank` is the recommended next diagnostic when the native Gaussian profile still renders broad ridges.  It changes the supervision contract rather than tuning another Gaussian loss coefficient:

- `models/deformable_detr_backbone.py` exposes ResNet `layer1` as an aux-only stride-4 feature.  The graph transformer continues to consume its original stride-8/16/32 levels.
- `AuxMapHead` fuses this actual stride-4 feature with the upsampled stride-8 decoder feature before its dedicated segmentation/direction towers and direct heatmap/offset projections.
- Positive focal cells come directly from graph-node coordinates (`AUX_HEATMAP_FOCAL_POS_SOURCE=node_centers`), and `val/heatmap_center_collision_rate` records nodes that collide in one native cell.
- Gaussian targets remain available only for negative weighting.  The profile disables MSE, ridge, local DSNT, variance, peakness, segmentation, and PAF terms so center focal + offset + ranking can be diagnosed in isolation.
- The ranking term removes every cell near any GT node from its negative set, then compares positives against normalized top-M hard foreground negatives with a distance-augmented margin.

The inference panel now shows native absolute probabilities separately from the segmentation-masked map; it no longer min-max normalizes diagnostic heatmaps.  `infer_aux_panel_treeformer.py` also writes `peak_pr_summary.json` for the requested native NMS threshold sweep.

Use this profile with a fresh stride-4 target cache and a checkpoint-free short run first:

```bash
PYTHONPATH=. .venv/bin/python train_hydra.py \
  train=seg_heatmap_paf +ablation=heatmap_center_offset_rank \
  DATA.DATASET=treeformer-2D DATA.DATA_PATH="$TREEFORMER_PRIVATE_DATA" \
  DATA.BATCH_SIZE=1 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 \
  DATA.TRAIN_LIMIT=2 DATA.VAL_LIMIT=2 TRAIN.EPOCHS=1 \
  checkpoint.enabled=false logging=disabled
```

## Runtime GPU Speedups

Hydra runtime config applies conservative GPU performance flags before model initialization:

- `runtime.cuda.allow_tf32=true`
- `runtime.cuda.cudnn_benchmark=true`
- `runtime.cuda.float32_matmul_precision=high`

These target Ampere-class GPUs and newer by allowing TensorFloat-32 matmul/convolution paths and cuDNN autotuning for fixed input sizes. Set `runtime.deterministic=true` for deterministic debugging; this disables cuDNN benchmark. Set `runtime.cuda.allow_tf32=false` or `runtime.cuda.float32_matmul_precision=highest` when exact FP32 behavior is more important than throughput.

`torch.compile` is intentionally opt-in and local:

- `runtime.compile.aux_head=true` compiles only the lightweight dense aux head with `nn.Module.compile()`. This preserves checkpoint `state_dict` keys.
- `runtime.compile.aux_loss=true` compiles the pure tensor aux-loss core. Target validation and range checks remain eager so bad masks still fail with clear errors.

For joint graph+aux training, FP16 AMP is also opt-in:

```bash
TREEFORMER_AMP=true \
TREEFORMER_AMP_DTYPE=float16 \
TREEFORMER_BATCH_SIZE=4 \
just train-private-joint-virtual-root-aux
```

AMP uses `GradScaler` and unscales before gradient clipping. Gradient
accumulation is not enabled by this switch. On the local RTX A4500, the
640x480, batch-4 smoke completed with FP16 and GPU EMA without OOM; confirm
VRAM again when changing the model, input size, or concurrent GPU workload.
- The full TreeFormer model is not compiled by default. The graph path still includes Python list/NestedTensor handling and a custom CUDA deformable-attention extension, both of which are poor first targets for whole-model fullgraph compilation.

The full segmentation recipe enables aux-head and aux-loss compile by default:

```bash
just train-private-seg-supervised
```

Disable either path without editing tracked config:

```bash
TREEFORMER_COMPILE_AUX_HEAD=false TREEFORMER_COMPILE_AUX_LOSS=false \
  just train-private-seg-supervised
```

The smoke recipe keeps compile disabled by default to avoid first-iteration compile overhead in short checks. Enable it explicitly when testing compile behavior:

```bash
TREEFORMER_COMPILE_AUX_HEAD=true TREEFORMER_COMPILE_AUX_LOSS=true \
  just smoke-private-seg-supervised
```

## Optimizers

`adamw_step` preserves the original AdamW + StepLR behavior.

`schedulefree_adamw` uses `schedulefree.AdamWScheduleFree`; training explicitly calls `optimizer.train()` before optimization and `optimizer.eval()` before validation/checkpointing.

`muon_schedulefree` uses the repository-local `MuonAdamW` optimizer as a base optimizer and wraps it with `schedulefree.ScheduleFreeWrapper`. Matrix-like hidden weights are assigned to Muon. Biases, normalization parameters, embeddings, heads, reference points, and sampling offsets are assigned to AdamW auxiliary groups. The assignment report is written to `${runtime.output_dir}/${runtime.parameter_report_name}`.

No implicit fallback is allowed. If `schedulefree` is missing or no parameter can be assigned to Muon, the run fails immediately.

## TensorBoard

When `logging=tensorboard`, rank 0 writes scalars to `${runtime.output_dir}/tensorboard`. The default smoke output root is `${TREEFORMER_ASSETS_ROOT}/trained_weights_hydra` unless `TREEFORMER_TRAIN_OUTPUT` or `TRAIN.SAVE_PATH` overrides it. The expected tags include:

- `train/total_loss`
- `train/class_loss`
- `train/nodes_loss`
- `train/edges_loss`
- `train/boxes_loss`
- `train/cards_loss`
- `val/smd`
- `optim/lr`
- `time/epoch_seconds`
- `checkpoint/best_metric`

For `train=aux_supervised`, expected tags instead include:

- `train/aux_total_loss`
- `train/seg_total_loss`
- `train/aux_seg_bce`
- `train/aux_seg_dice_loss`
- `train/aux_seg_focal_loss`
- `train/aux_heatmap_mse`
- `train/aux_paf_l1`
- `train/aux_heatmap_coord_loss`, `train/aux_heatmap_coord_var_loss`, `train/aux_heatmap_peak_loss`
- `train/aux_paf_total_loss`, `train/aux_paf_angular`
- `val/aux_total_loss`
- `val/seg_total_loss`
- `val/seg_iou`
- `val/seg_dice_score`
- `val/seg_soft_dice_score`
- `val/seg_precision`
- `val/seg_recall`
- `val/pred_positive_rate`
- `val/target_positive_rate`
- `val/heatmap_mae`
- `val/paf_masked_l1`
- `val/aux_heatmap_coord_loss`, `val/aux_heatmap_coord_var_loss`, `val/aux_heatmap_peak_loss`, `val/aux_paf_angular`
- `val/heatmap_node_recall`, `val/heatmap_node_precision`, `val/heatmap_duplicate_peak_rate`, `val/heatmap_background_peaks_per_image`

## EMA and checkpointing

`ema=default` enables EMA updates after every optimizer step. When `ema.evaluate=true`, validation runs under EMA weights and then restores live model weights. For newly initialized aux heads, prefer `ema.evaluate=false` until enough updates have accumulated; otherwise validation and qualitative panels can reflect mostly-initial EMA weights.

Checkpoints are written by `CheckpointManager`:

- `last.pt`: updated on validation epochs when enabled. With `TRAIN.VAL_INTERVAL>1`, the latest non-validation epoch is not checkpointed; use a smaller interval if interruption-resume granularity matters.
- `best.pt`: updated only when `checkpoint.metric_name` improves according to `checkpoint.mode`.
- `epoch_000000.pt`: periodic checkpoint when `checkpoint.save_every > 0`.

Each checkpoint stores model, optimizer, scheduler, metrics, EMA state, and resolved Hydra config.

## Post-training inference panels

After a stage finishes, render per-image summary panels from `best.pt` before deciding whether to keep the stage. The renderer reads current Hydra checkpoints, including embedded resolved config and EMA shadow weights. With `--weights auto`, EMA weights are preferred because validation also runs under EMA when `ema.evaluate=true`. For `train=seg_supervised` and `train=aux_supervised` runs using `ema.evaluate=false`, render aux panels with `--weights model`; the `just infer-aux-panels` recipe defaults to that behavior.

For a legacy TreeFormer-format dataset, keep the concrete dataset root in the environment:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
export TREEFORMER_INFER_CHECKPOINT=<stage_best_checkpoint>
export TREEFORMER_INFER_OUTPUT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/inference_panels/<stage_name>

just infer-panels
```

To combine graph overlays with dense diagnostics in the same summary panel, pass an aux-supervised checkpoint separately. This is useful when the graph stage was trained without an aux head, but you still want to inspect segmentation, node heatmap, and PAF / edge-direction maps beside the graph result:

```bash
export TREEFORMER_INFER_CHECKPOINT=<graph_stage_best_checkpoint>
export TREEFORMER_INFER_AUX_CHECKPOINT=<aux_stage_best_checkpoint>
export TREEFORMER_INFER_OUTPUT=${TREEFORMER_ASSETS_ROOT:-../TreeFormer_assets}/inference_panels_with_aux/<stage_name>
export TREEFORMER_MAX_SIZE=512

just infer-panels-with-aux
```

`infer-panels-with-aux` appends GT / predicted segmentation overlays, GT / predicted node heatmaps, and GT / predicted edge-direction maps to each graph summary image. Heatmap or PAF prediction panels are hidden automatically when the aux checkpoint recorded a zero loss weight for that target; pass `--aux-show-untrained-maps` only when debugging raw, untrained channels.

For an aux-enabled graph checkpoint, `infer_panel_treeformer.py` also gates graph-token candidates with the **predicted** segmentation confidence before relation / MST postprocessing. The default threshold is `0.5`; this path never reads a GT mask. The graph nodes, heatmap display, and edge-direction display therefore share the model's own foreground decision at inference. A graph checkpoint without an auxiliary segmentation head must opt out explicitly with `--graph-node-segmentation-threshold 0`.

The recipe renders the validation split as:

```bash
PYTHONPATH=. "$TREEFORMER_PYTHON" infer_panel_treeformer.py \
  --legacy-split-root "$TREEFORMER_PRIVATE_DATA/val" \
  --output-dir "$TREEFORMER_INFER_OUTPUT" \
  --run "Ours|$TREEFORMER_INFER_CHECKPOINT|mst" \
  --device cuda \
  --max-size 128 \
  --inset \
  --save-graph-json
```

Each input image gets `<sample_id>_panel.png`; `--save-graph-json` also writes `<sample_id>_pred_graph.json`. Use `TREEFORMER_INFER_MODE=raw` for unconstrained relation output, or `TREEFORMER_INFER_MODE=mst-dist` for the distance-weighted MST variant. For a quick sanity check before rendering a full split, call `infer_panel_treeformer.py` directly with `--limit 3`. Keep output under `${TREEFORMER_ASSETS_ROOT}` and do not commit generated panels or graph JSON.
