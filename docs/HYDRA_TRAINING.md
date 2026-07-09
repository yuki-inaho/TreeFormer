# Hydra Training Infrastructure

This repository now includes a Hydra-managed training entrypoint for TreeFormer experiments.

## Main entrypoint

```bash
export TREEFORMER_PYTHON=${TREEFORMER_PYTHON:-../venv/TreeFormer/bin/python}
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

- `train=seg_only`: segmentation loss only. It treats segmentation as background + one foreground stem class, sets detail / heatmap / PAF loss weights to zero, and checkpoints on `val/seg_soft_dice_score`.
- `train=seg_heatmap`: segmentation plus node heatmap. It keeps graph output and graph losses disabled, adds `W_AUX_HEATMAP=1`, keeps PAF loss at zero, and uses `DATA.AUX_TARGET_MODE=seg_heatmap` so `FastSegSupervisedDataset` generates node heatmaps from the split graph annotation.
- `train=seg_supervised`: legacy compatibility mode from the earlier stabilization work. It is segmentation plus weak external-mask-derived detail boundary regularization.

The segmentation target must come from the split-local external TPE binary mask directory, `seg/` preferred and `unet/` accepted only as a legacy fallback. The graph-derived raster mask is not used as the segmentation target in these modes.

Config-only check:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just cfg-private-seg-only
just cfg-private-seg-heatmap
just cfg-private-seg-supervised
just cfg-private-aux-supervised
```

Short GPU smoke:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just smoke-private-seg-only
just smoke-private-seg-heatmap
just smoke-private-seg-supervised
just smoke-private-aux-supervised
```

Full no-geometry aux stage:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just cache-private-fast-seg
just train-private-seg-only
just train-private-seg-heatmap
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

Operational notes:

- Use `augmentation=disabled` first. Do not use geometric, deformation, crop, rotate, affine, perspective, or elastic DA for this stage.
- `checkpoint.pretrained_strict=false` is expected because the pretrained graph model has no aux head parameters.
- `DATA.SEGMENTATION_TARGET_SOURCE=external_mask` is expected for `train=seg_supervised` and `train=aux_supervised`. Missing `seg/<sample_id>.png` or legacy `unet/<sample_id>.png` is a configuration error, not a reason to fall back silently to graph-derived labels.
- External mask images may be stored as `0/255` PNGs; the loader thresholds them to binary float targets before BCE / Dice / Focal loss, and the loss code rejects targets outside `[0, 1]`.
- The default segmentation loss uses binary BCE + Dice with Focal disabled. `AUX_SEG_POS_WEIGHT=auto` is capped conservatively so rare foreground does not cause broad positive overprediction.
- `train=seg_only` and `train=seg_heatmap` use `FastSegSupervisedDataset` by default. `DATA.AUX_TARGET_MODE=seg_only` returns zero-valued unused PAF/heatmap tensors. `DATA.AUX_TARGET_MODE=seg_heatmap` generates only node heatmap targets and leaves PAF tensors zero.
- `train=seg_supervised` adds a weak optional STDC-style detail boundary auxiliary head as the fifth aux channel. Its target is a multi-scale Laplacian boundary map derived from the external segmentation mask, not a graph edge or PAF. `W_AUX_DETAIL=0.1` keeps it as a light boundary-sharpening regularizer while BCE + Dice segmentation remains the primary objective.
- For full segmentation-only training, generate repo-external cache first with `just cache-private-fast-seg`, then run `just train-private-seg-supervised`. The full recipe uses `DATA.SEG_CACHE_MODE=disk`, `DATA.NUM_WORKERS=4`, `DATA.PERSISTENT_WORKERS=true`, and `DATA.PREFETCH_FACTOR=2` by default. Set `TREEFORMER_SEG_CACHE_MODE=none` to bypass disk cache.
- `FastSegSupervisedDataset` defaults to `DATA.SEG_RESIZE_POLICY=legacy_half`, matching the legacy loader behavior that halves the raw image before applying `DATA.MAX_SIZE`. For a raw 800x600 image, `DATA.MAX_SIZE=128` becomes 128x96 and `DATA.MAX_SIZE=512` still caps at 400x300. Use `DATA.SEG_RESIZE_POLICY=full` with `DATA.MAX_SIZE=512` to train at 512x384 from the original image.
- Aux/seg stages may keep EMA updates enabled, but use `ema.evaluate=false` for validation and best-checkpoint selection because the aux head is newly initialized and `decay=0.9999` changes too slowly during short stages.
- To initialize a dense stage from a previous Hydra `best.pt`, set `TREEFORMER_PRETRAINED_CHECKPOINT=<best.pt>` and `TREEFORMER_PRETRAINED_KEY=model`. Fork-source checkpoints keep the default `TREEFORMER_PRETRAINED_KEY=net`.
- RGB input is normalized by the legacy loader with `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`, so model input is scaled from `[0, 1]` to `[-1, 1]`.
- In `train=seg_only` and `train=seg_supervised`, `W_AUX_HEATMAP=0` and `W_AUX_PAF=0`; heatmap and PAF channels are not trained in those stages. In `train=seg_heatmap`, `W_AUX_HEATMAP=1` and `W_AUX_PAF=0`. The aux output layout is segmentation, heatmap, PAF-x, PAF-y, optional detail boundary. `train=aux_supervised` keeps the original 4-channel layout and has `W_AUX_DETAIL=0`.
- Keep `DATA.BATCH_SIZE=12` and `DATA.MAX_SIZE=128` unless the GPU is shared or memory pressure is observed.
- Monitor `val/seg_soft_dice_score`, `val/seg_dice_score`, `val/seg_iou`, `val/seg_precision`, `val/seg_recall`, and `val/pred_positive_rate`; do not interpret `val/smd` for this mode because graph validation is intentionally skipped. `val/seg_dice_score` and IoU are hard-threshold metrics at `AUX_SEG_THRESHOLD`; early runs may show zero while soft Dice and loss still improve.

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
| S0. Segmentation | `train=seg_only augmentation=disabled` | 20 | `1e-4` / `3e-5` | Confirm the RGB encoder can learn the external TPE binary stem mask before adding heatmap / PAF / graph objectives. | Continue only if `val/seg_soft_dice_score` or `val/seg_total_loss` improves, then check hard-threshold Dice/IoU and foreground rate for calibration. |
| H0. Seg + Heatmap | `train=seg_heatmap augmentation=disabled` | 20 | `1e-4` / `3e-5` | Add node heatmap supervision while keeping graph output and PAF disabled. | Continue only if heatmap error declines without collapsing segmentation metrics. |
| A0. Aux maps | `train=aux_supervised augmentation=disabled` | 20 | `1e-4` / `3e-5` | Add PAF direction supervision after segmentation and heatmap are learnable. | Continue only if `val/aux_total_loss` declines without collapsing segmentation metrics. |
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
| `train` | `default`, `dry_run`, `seg_supervised`, `aux_supervised` | Epochs, loss weights, LR, save path, graph-vs-aux training mode |
| `optimizer` | `adamw_step`, `schedulefree_adamw`, `muon_schedulefree` | Optimizer and scheduler selection |
| `logging` | `tensorboard`, `disabled` | TensorBoard event writing |
| `ema` | `disabled`, `default` | EMA update/evaluation behavior |
| `checkpoint` | `default` | last/best/periodic checkpoint policy |
| `distributed` | `single`, `ddp` | Single-process or torchrun/DDP execution |

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

## Runtime GPU Speedups

Hydra runtime config applies conservative GPU performance flags before model initialization:

- `runtime.cuda.allow_tf32=true`
- `runtime.cuda.cudnn_benchmark=true`
- `runtime.cuda.float32_matmul_precision=high`

These target Ampere-class GPUs and newer by allowing TensorFloat-32 matmul/convolution paths and cuDNN autotuning for fixed input sizes. Set `runtime.deterministic=true` for deterministic debugging; this disables cuDNN benchmark. Set `runtime.cuda.allow_tf32=false` or `runtime.cuda.float32_matmul_precision=highest` when exact FP32 behavior is more important than throughput.

`torch.compile` is intentionally opt-in and local:

- `runtime.compile.aux_head=true` compiles only the lightweight dense aux head with `nn.Module.compile()`. This preserves checkpoint `state_dict` keys.
- `runtime.compile.aux_loss=true` compiles the pure tensor aux-loss core. Target validation and range checks remain eager so bad masks still fail with clear errors.
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

## EMA and checkpointing

`ema=default` enables EMA updates after every optimizer step. When `ema.evaluate=true`, validation runs under EMA weights and then restores live model weights. For newly initialized aux heads, prefer `ema.evaluate=false` until enough updates have accumulated; otherwise validation and qualitative panels can reflect mostly-initial EMA weights.

Checkpoints are written by `CheckpointManager`:

- `last.pt`: always updated when enabled.
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
