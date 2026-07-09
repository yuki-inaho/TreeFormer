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

- segmentation logit
- node heatmap logit
- PAF x direction
- PAF y direction

The training loss is:

- `train/aux_seg_bce`: BCE-with-logits against the dataloader `unet` mask
- `train/aux_heatmap_mse`: MSE between sigmoid heatmap output and generated node heatmap
- `train/aux_paf_l1`: masked L1 between tanh PAF output and generated PAF vectors

Validation uses the same direct supervision and checkpoints on `val/aux_total_loss`. Graph losses are not computed, and the graph SMD validator is skipped. This makes the first question concrete: can the network learn the mask / heatmap / direction fields from RGB before asking it to output a clean graph.

Config-only check:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just cfg-private-aux-supervised
```

Short GPU smoke:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just smoke-private-aux-supervised
```

Full no-geometry aux stage:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just train-private-aux-supervised
```

Operational notes:

- Use `augmentation=disabled` first. Do not use geometric, deformation, crop, rotate, affine, perspective, or elastic DA for this stage.
- `checkpoint.pretrained_strict=false` is expected because the pretrained graph model has no aux head parameters.
- Keep `DATA.BATCH_SIZE=12` and `DATA.MAX_SIZE=128` unless the GPU is shared or memory pressure is observed.
- Monitor `val/aux_total_loss`, `val/seg_iou`, `val/heatmap_mae`, and `val/paf_masked_l1`; do not interpret `val/smd` for this mode because graph validation is intentionally skipped.

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
| A0. Aux maps | `train=aux_supervised augmentation=disabled` | 20 | `1e-4` / `3e-5` | Confirm the RGB encoder can learn segmentation, node heatmap, and PAF direction fields from direct supervision. | Continue only if `val/aux_total_loss` declines and `val/seg_iou` improves. |
| G0. Graph stabilize | `augmentation=disabled` | 20 | `3e-5` / `1e-5` | Re-enable graph output only after aux maps show learnable supervision. | Continue only if train loss declines and `val/smd` does not spike. |
| G1. Optical | `augmentation=photometric_opencv` | 80 | `5e-5` / `1.5e-5` | Learn robustness to illumination, noise, blur, and color shifts while graph labels stay unchanged. | Use best checkpoint if `val/smd` improves; otherwise return to previous best. |

Operational notes:

- Use `best.pt` from the previous stage as `checkpoint.resume` for the next stage, and set `checkpoint.pretrained=null` after Stage 0.
- Keep `DATA.BATCH_SIZE=12`, `DATA.MAX_SIZE=128`, GPU EMA, TensorBoard, and Muon + ScheduleFree unless a stage shows instability.
- Do not use random crop, rotate, scale, affine, perspective, elastic, or graph-deformation augmentation in this curriculum.
- Prefer the OpenCV photometric config for Stage 1. Use `regularized_albumentationsx` only after explicit license acceptance and optional dependency installation, and only for image-only photometric experiments.
- Do not mix private dataset paths into docs, commit messages, or work records; pass them through `TREEFORMER_PRIVATE_DATA`.

Config-only checks are available:

```bash
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
| `train` | `default`, `dry_run`, `aux_supervised` | Epochs, loss weights, LR, save path, graph-vs-aux training mode |
| `optimizer` | `adamw_step`, `schedulefree_adamw`, `muon_schedulefree` | Optimizer and scheduler selection |
| `logging` | `tensorboard`, `disabled` | TensorBoard event writing |
| `ema` | `disabled`, `default` | EMA update/evaluation behavior |
| `checkpoint` | `default` | last/best/periodic checkpoint policy |
| `distributed` | `single`, `ddp` | Single-process or torchrun/DDP execution |

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
- `train/aux_seg_bce`
- `train/aux_heatmap_mse`
- `train/aux_paf_l1`
- `val/aux_total_loss`
- `val/seg_iou`
- `val/heatmap_mae`
- `val/paf_masked_l1`

## EMA and checkpointing

`ema=default` enables EMA updates after every optimizer step. When `ema.evaluate=true`, validation runs under EMA weights and then restores live model weights.

Checkpoints are written by `CheckpointManager`:

- `last.pt`: always updated when enabled.
- `best.pt`: updated only when `checkpoint.metric_name` improves according to `checkpoint.mode`.
- `epoch_000000.pt`: periodic checkpoint when `checkpoint.save_every > 0`.

Each checkpoint stores model, optimizer, scheduler, metrics, EMA state, and resolved Hydra config.

## Post-training inference panels

After a stage finishes, render per-image summary panels from `best.pt` before deciding whether to keep the stage. The renderer reads current Hydra checkpoints, including embedded resolved config and EMA shadow weights. With `--weights auto`, EMA weights are preferred because validation also runs under EMA when `ema.evaluate=true`.

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
