set dotenv-load := false

python := env_var_or_default("TREEFORMER_PYTHON", "../venv/TreeFormer/bin/python")
assets_root := env_var_or_default("TREEFORMER_ASSETS_ROOT", "../TreeFormer_assets")
private_treeformer_data := env_var_or_default("TREEFORMER_PRIVATE_DATA", "")
fork_grapevine_pretrained := env_var_or_default("TREEFORMER_PRETRAINED_CHECKPOINT", assets_root + "/pretrained_weights/fork_source_main/grapevein/checkpoint_ours.pkl")
pretrained_key := env_var_or_default("TREEFORMER_PRETRAINED_KEY", "net")
private_pretrained_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT", assets_root + "/trained_weights_hydra_private_pretrained")
private_pretrained_aug_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_AUG", assets_root + "/trained_weights_hydra_private_pretrained_aug")
private_pretrained_curriculum_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_CURRICULUM", assets_root + "/trained_weights_hydra_private_curriculum")
private_aux_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_AUX", assets_root + "/trained_weights_hydra_aux_supervised")
private_seg_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_SEG", assets_root + "/trained_weights_hydra_seg_supervised")
private_seg_only_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_SEG_ONLY", assets_root + "/trained_weights_hydra_seg_only")
private_seg_heatmap_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_SEG_HEATMAP", assets_root + "/trained_weights_hydra_seg_heatmap")
private_seg_heatmap_paf_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_SEG_HEATMAP_PAF", assets_root + "/trained_weights_hydra_seg_heatmap_paf")
seg_cache_root := env_var_or_default("TREEFORMER_SEG_CACHE_ROOT", assets_root + "/cache/fast_seg/private_seg_max128")
aux_panel_output := env_var_or_default("TREEFORMER_AUX_PANEL_OUTPUT", assets_root + "/aux_inference_panels")

hydra-cfg:
    PYTHONPATH=. {{python}} train_hydra.py --cfg job

test:
    PYTHONPATH=. {{python}} -m pytest -q

lint:
    PYTHONPATH=. {{python}} -m ruff check .

format:
    PYTHONPATH=. {{python}} -m ruff format .

install-albumentationsx:
    uv pip install --python {{python}} --project . --group albumentationsx

smoke-hydra-cpu-dry-run:
    PYTHONPATH=. {{python}} train_hydra.py train=dry_run runtime.device=cpu runtime.fail_if_cuda_unavailable=false logging=disabled checkpoint.enabled=false

smoke-hydra-gpu:
    PYTHONPATH=. {{python}} train_hydra.py optimizer=muon_schedulefree ema=default TRAIN.EPOCHS=1 DATA.TRAIN_LIMIT=1 DATA.VAL_LIMIT=1 DATA.BATCH_SIZE=1

cfg-private-pretrained-gpu-batch12:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job optimizer=muon_schedulefree ema=default logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net checkpoint.pretrained_strict=true DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_pretrained_output}}

smoke-private-pretrained-gpu-batch12:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py optimizer=muon_schedulefree ema=default logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net checkpoint.pretrained_strict=true DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_pretrained_output}}

cfg-private-pretrained-gpu-batch12-aug:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job augmentation=regularized optimizer=muon_schedulefree ema=default logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net checkpoint.pretrained_strict=true DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_pretrained_aug_output}} log.exp_name=private_pretrained_aug_batch12

smoke-private-pretrained-gpu-batch12-aug:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py augmentation=regularized optimizer=muon_schedulefree ema=default logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net checkpoint.pretrained_strict=true DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_pretrained_aug_output}} TRAIN.EPOCHS=3 log.exp_name=private_pretrained_aug_batch12

infer-panels:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @test -n "${TREEFORMER_INFER_CHECKPOINT:-}" || (echo "Set TREEFORMER_INFER_CHECKPOINT to best.pt or another checkpoint" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} infer_panel_treeformer.py --legacy-split-root "{{private_treeformer_data}}/val" --output-dir "${TREEFORMER_INFER_OUTPUT:-{{assets_root}}/inference_panels}" --run "${TREEFORMER_INFER_LABEL:-Ours}|${TREEFORMER_INFER_CHECKPOINT}|${TREEFORMER_INFER_MODE:-mst}" --device cuda --max-size 128 --inset --save-graph-json

infer-aux-panels:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @test -n "${TREEFORMER_AUX_CHECKPOINT:-}" || (echo "Set TREEFORMER_AUX_CHECKPOINT to an aux-supervised best.pt or last.pt checkpoint" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} infer_aux_panel_treeformer.py --legacy-split-root "{{private_treeformer_data}}/val" --checkpoint "${TREEFORMER_AUX_CHECKPOINT}" --output-dir "{{aux_panel_output}}" --device cuda --max-size 128 --limit "${TREEFORMER_AUX_PANEL_LIMIT:-10}" --weights "${TREEFORMER_AUX_WEIGHTS:-model}" --save-json

cfg-private-curriculum-stage0:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job augmentation=disabled optimizer=muon_schedulefree ema=default logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net checkpoint.pretrained_strict=true DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_pretrained_curriculum_output}} TRAIN.EPOCHS=20 TRAIN.LR=3e-5 TRAIN.LR_BACKBONE=1e-5 log.exp_name=private_curriculum_stage0_stabilize

cfg-private-curriculum-stage1:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job augmentation=photometric_opencv optimizer=muon_schedulefree ema=default logging=tensorboard checkpoint.pretrained=null checkpoint.resume="${TREEFORMER_CURRICULUM_RESUME:?Set TREEFORMER_CURRICULUM_RESUME to the previous stage checkpoint}" DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_pretrained_curriculum_output}} TRAIN.EPOCHS=80 TRAIN.LR=5e-5 TRAIN.LR_BACKBONE=1.5e-5 log.exp_name=private_curriculum_stage1_photometric

cfg-private-aux-supervised:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=aux_supervised augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_aux_output}} TRAIN.EPOCHS=20 TRAIN.LR=1e-4 TRAIN.LR_BACKBONE=3e-5 log.exp_name=private_aux_supervised_no_da

smoke-private-aux-supervised:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=aux_supervised augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_aux_output}} TRAIN.EPOCHS=3 TRAIN.LR=1e-4 TRAIN.LR_BACKBONE=3e-5 log.exp_name=private_aux_supervised_smoke_no_da

train-private-aux-supervised:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=aux_supervised augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE=128 DATA.NUM_WORKERS=0 DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_aux_output}} TRAIN.EPOCHS=20 TRAIN.LR=1e-4 TRAIN.LR_BACKBONE=3e-5 log.exp_name=private_aux_supervised_no_da

cache-private-fast-seg:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} generate_fast_seg_cache.py --dataset-root "{{private_treeformer_data}}" --cache-root "{{seg_cache_root}}" --splits train val --max-size "${TREEFORMER_MAX_SIZE:-128}" --resize-policy "${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" --aux-target-mode "${TREEFORMER_AUX_TARGET_MODE:-seg_only}" --heatmap-sigma "${TREEFORMER_HEATMAP_SIGMA:-3.0}" --heatmap-cutoff "${TREEFORMER_HEATMAP_CUTOFF:-0.01}"

cache-private-fast-seg-heatmap:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} generate_fast_seg_cache.py --dataset-root "{{private_treeformer_data}}" --cache-root "{{seg_cache_root}}" --splits train val --max-size "${TREEFORMER_MAX_SIZE:-128}" --resize-policy "${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" --aux-target-mode seg_heatmap --heatmap-sigma "${TREEFORMER_HEATMAP_SIGMA:-3.0}" --heatmap-cutoff "${TREEFORMER_HEATMAP_CUTOFF:-0.01}"

cache-private-fast-seg-heatmap-paf:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} generate_fast_seg_cache.py --dataset-root "{{private_treeformer_data}}" --cache-root "{{seg_cache_root}}" --splits train val --max-size "${TREEFORMER_MAX_SIZE:-128}" --resize-policy "${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" --aux-target-mode seg_heatmap_paf --heatmap-sigma "${TREEFORMER_HEATMAP_SIGMA:-3.0}" --heatmap-cutoff "${TREEFORMER_HEATMAP_CUTOFF:-0.01}" --paf-line-thickness "${TREEFORMER_PAF_LINE_THICKNESS:-2}" --paf-mask-thickness "${TREEFORMER_PAF_MASK_THICKNESS:-6}"

cfg-private-seg-supervised:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=seg_supervised augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_supervised_fast_no_da}"

smoke-private-seg-supervised:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_supervised augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-none}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_seg_output}} TRAIN.EPOCHS=3 TRAIN.LR=1e-4 TRAIN.LR_BACKBONE=3e-5 runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name=private_seg_supervised_smoke_no_da

train-private-seg-supervised:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_supervised augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key=net DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_supervised_fast_no_da}"

cfg-private-seg-only:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=seg_only augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_only_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_only_no_da}"

smoke-private-seg-only:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_only augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-none}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_seg_only_output}} TRAIN.EPOCHS=3 TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name=private_seg_only_smoke_no_da

train-private-seg-only:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_only augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_only_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_only_no_da}"

cfg-private-seg-heatmap:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=seg_heatmap augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_heatmap_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_heatmap_no_da}"

smoke-private-seg-heatmap:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_heatmap augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-none}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_seg_heatmap_output}} TRAIN.EPOCHS=3 TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name=private_seg_heatmap_smoke_no_da

train-private-seg-heatmap:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_heatmap augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_heatmap_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_heatmap_no_da}"

cfg-private-seg-heatmap-paf:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=seg_heatmap_paf augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_heatmap_paf_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_heatmap_paf_no_da}"

smoke-private-seg-heatmap-paf:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_heatmap_paf augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-none}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_seg_heatmap_paf_output}} TRAIN.EPOCHS=3 TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name=private_seg_heatmap_paf_smoke_no_da

train-private-seg-heatmap-paf:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_heatmap_paf augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE=12 DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-128}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_seg_heatmap_paf_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-true}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-true}" log.exp_name="${TREEFORMER_EXP_NAME:-private_seg_heatmap_paf_no_da}"
