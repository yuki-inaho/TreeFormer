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
just cfg-private-pretrained-gpu-batch12-aug
just smoke-private-pretrained-gpu-batch12-aug
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

## Augmentation

Use `augmentation=regularized` for the reusable DA path. It is disabled by default to preserve the legacy baseline.

The augmentation code is split by contract:

- photometric transforms are image-only and leave graph nodes / edges unchanged.
- affine and elastic transforms update the RGB image and normalized node coordinates from the same transform field.
- edge topology is preserved; transforms that would move any node outside the image are rejected for that sample.

`augmentation=regularized` also sets `DATA.LEGACY_ROTATE=false`, so geometry regularization is owned by the graph-aware augmentation module instead of stacking on top of the legacy always-rotate path.

Photometric backend selection:

- `backend=albumentationsx` tries to import the AlbumentationsX-compatible `albumentations` module.
- `allow_fallback=true` falls back to the OpenCV implementation when AlbumentationsX is unavailable.
- `backend=opencv` uses the portable in-repo OpenCV implementation only.

AlbumentationsX is optional because it has a separate dual-license model and can require platform-specific binary dependencies. To enable that backend in the active venv:

```bash
uv pip install --python "$TREEFORMER_PYTHON" 'albumentationsx>=2.3,<3.0'
```

For a short GPU smoke run with pretrained weights and DA:

```bash
export TREEFORMER_PRIVATE_DATA=<legacy_treeformer_dataset_root>
just cfg-private-pretrained-gpu-batch12-aug
just smoke-private-pretrained-gpu-batch12-aug
```

The smoke recipe runs 3 epochs with batch size 12, GPU EMA, TensorBoard, Muon + ScheduleFree, and repo-external output paths. Keep the dataset root in the environment variable and out of committed docs.

## Config groups

| Group | Files | Purpose |
|---|---|---|
| `data` | `guyot_smoke`, `guyot_full` | Guyot dataset paths, limits, workers, seed |
| `augmentation` | `disabled`, `regularized` | Image-only photometric DA and graph-aware affine / elastic DA |
| `model` | `treeformer_2d` | Existing 2D TreeFormer architecture settings |
| `train` | `default`, `dry_run` | Epochs, loss weights, LR, save path |
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

## EMA and checkpointing

`ema=default` enables EMA updates after every optimizer step. When `ema.evaluate=true`, validation runs under EMA weights and then restores live model weights.

Checkpoints are written by `CheckpointManager`:

- `last.pt`: always updated when enabled.
- `best.pt`: updated only when `checkpoint.metric_name` improves according to `checkpoint.mode`.
- `epoch_000000.pt`: periodic checkpoint when `checkpoint.save_every > 0`.

Each checkpoint stores model, optimizer, scheduler, metrics, EMA state, and resolved Hydra config.
