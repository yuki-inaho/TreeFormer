set dotenv-load := false

python := env_var_or_default("TREEFORMER_PYTHON", ".venv/bin/python")
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
private_ablation_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_ABLATION", assets_root + "/trained_weights_hydra_aux_ablation")
private_joint_vr_aux_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_JOINT_VR_AUX", assets_root + "/trained_weights_hydra_joint_virtual_root_aux")
private_optuna_joint_vr_aux_output := env_var_or_default("TREEFORMER_OPTUNA_OUTPUT", assets_root + "/optuna/joint_virtual_root_aux")
seg_cache_root := env_var_or_default("TREEFORMER_SEG_CACHE_ROOT", assets_root + "/cache/fast_seg/private_seg_max128")
joint_optuna_cache_root := env_var_or_default("TREEFORMER_JOINT_OPTUNA_CACHE_ROOT", assets_root + "/cache/fast_seg/joint_virtual_root_aux_v2")
aux_panel_output := env_var_or_default("TREEFORMER_AUX_PANEL_OUTPUT", assets_root + "/aux_inference_panels")
ablation_name := env_var_or_default("TREEFORMER_ABLATION", "heatmap_mse_baseline")

setup-venv:
    # Order matters here and must not be reversed: uv sync runs BEFORE the CUDA ops
    # build, and always with --inexact. If a bare `uv sync --all-groups` runs after
    # the build, it deletes the already-built MultiScaleDeformableAttention .so
    # (verified: it reports "Would uninstall 1 package -
    # multiscaledeformableattention==1.0" because the build's `setup.py install`
    # registers an .egg-info that uv treats as a removable distribution). Running
    # sync first means there is nothing built yet to delete, so this order is a
    # structural defense against forgetting --inexact later, not just a convention.
    uv venv --python 3.10
    uv sync --inexact --all-groups
    (cd models/ops && PATH=/usr/local/cuda/bin:$PATH ../../.venv/bin/python setup.py build install)
    PYTHONPATH=. .venv/bin/python -c "import torch, MultiScaleDeformableAttention; from models.ops.modules import MSDeformAttn; assert torch.cuda.is_available(), 'GPU inference is required but CUDA is unavailable'; print('MSDA OK', torch.__version__, torch.cuda.get_device_name(0))"

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

install-optuna:
    uv pip install --python {{python}} --project . --group tuning

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

infer-panels-with-aux:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @test -n "${TREEFORMER_INFER_CHECKPOINT:-}" || (echo "Set TREEFORMER_INFER_CHECKPOINT to graph best.pt or another checkpoint" >&2; exit 2)
    @test -n "${TREEFORMER_INFER_AUX_CHECKPOINT:-${TREEFORMER_AUX_CHECKPOINT:-}}" || (echo "Set TREEFORMER_INFER_AUX_CHECKPOINT or TREEFORMER_AUX_CHECKPOINT to an aux-supervised checkpoint" >&2; exit 2)
    @AUX_CHECKPOINT="${TREEFORMER_INFER_AUX_CHECKPOINT:-${TREEFORMER_AUX_CHECKPOINT:-}}"; PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} infer_panel_treeformer.py --legacy-split-root "{{private_treeformer_data}}/val" --output-dir "${TREEFORMER_INFER_OUTPUT:-{{assets_root}}/inference_panels_with_aux}" --run "${TREEFORMER_INFER_LABEL:-Ours}|${TREEFORMER_INFER_CHECKPOINT}|${TREEFORMER_INFER_MODE:-mst}" --device cuda --max-size "${TREEFORMER_MAX_SIZE:-128}" --inset --save-graph-json --include-aux-maps --aux-checkpoint "$AUX_CHECKPOINT" --aux-weights "${TREEFORMER_INFER_AUX_WEIGHTS:-model}" --aux-loader "${TREEFORMER_INFER_AUX_LOADER:-auto}" --aux-max-size "${TREEFORMER_AUX_MAX_SIZE:-${TREEFORMER_MAX_SIZE:-128}}" --columns "${TREEFORMER_INFER_COLUMNS:-3}" --panel-width "${TREEFORMER_PANEL_WIDTH:-360}"

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

cache-private-fast-seg-heatmap-paf-ablation:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} generate_fast_seg_cache.py --dataset-root "{{private_treeformer_data}}" --cache-root "{{seg_cache_root}}" --splits train val --max-size "${TREEFORMER_MAX_SIZE:-128}" --resize-policy "${TREEFORMER_SEG_RESIZE_POLICY:-legacy_half}" --aux-target-mode seg_heatmap_paf --heatmap-sigma "${TREEFORMER_HEATMAP_SIGMA:-3.0}" --heatmap-cutoff "${TREEFORMER_HEATMAP_CUTOFF:-0.01}" --paf-line-thickness "${TREEFORMER_PAF_LINE_THICKNESS:-2}" --paf-mask-thickness "${TREEFORMER_PAF_MASK_THICKNESS:-6}"

cache-private-joint-virtual-root-aux-optuna:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} generate_fast_seg_cache.py --dataset-root "{{private_treeformer_data}}" --cache-root "{{joint_optuna_cache_root}}/heatmap_sigma_3_0" --splits train val --max-size "${TREEFORMER_MAX_SIZE:-640}" --resize-policy full --aux-target-mode seg_heatmap_paf --heatmap-sigma 3.0 --heatmap-cutoff "${TREEFORMER_HEATMAP_CUTOFF:-0.01}" --paf-line-thickness "${TREEFORMER_PAF_LINE_THICKNESS:-2}" --paf-mask-thickness "${TREEFORMER_PAF_MASK_THICKNESS:-6}"
    @PYTHONPATH=. {{python}} generate_fast_seg_cache.py --dataset-root "{{private_treeformer_data}}" --cache-root "{{joint_optuna_cache_root}}/heatmap_sigma_1_5" --splits train val --max-size "${TREEFORMER_MAX_SIZE:-640}" --resize-policy full --aux-target-mode seg_heatmap_paf --heatmap-sigma 1.5 --heatmap-cutoff "${TREEFORMER_HEATMAP_CUTOFF:-0.01}" --paf-line-thickness "${TREEFORMER_PAF_LINE_THICKNESS:-2}" --paf-mask-thickness "${TREEFORMER_PAF_MASK_THICKNESS:-6}"

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

cfg-private-seg-heatmap-paf-ablation:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=seg_heatmap_paf +ablation={{ablation_name}} augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE="${TREEFORMER_BATCH_SIZE:-8}" DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-640}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-full}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_ablation_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name="${TREEFORMER_EXP_NAME:-{{ablation_name}}}"

smoke-private-seg-heatmap-paf-ablation:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_heatmap_paf +ablation={{ablation_name}} augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE="${TREEFORMER_BATCH_SIZE:-8}" DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-640}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-none}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-full}" DATA.TRAIN_LIMIT=24 DATA.VAL_LIMIT=12 TRAIN.SAVE_PATH={{private_ablation_output}} TRAIN.EPOCHS=3 TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name="${TREEFORMER_EXP_NAME:-{{ablation_name}}_smoke}"

train-private-seg-heatmap-paf-ablation:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=seg_heatmap_paf +ablation={{ablation_name}} augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE="${TREEFORMER_BATCH_SIZE:-8}" DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-640}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-4}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY="${TREEFORMER_SEG_RESIZE_POLICY:-full}" DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_ablation_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" runtime.compile.aux_head="${TREEFORMER_COMPILE_AUX_HEAD:-false}" runtime.compile.aux_loss="${TREEFORMER_COMPILE_AUX_LOSS:-false}" log.exp_name="${TREEFORMER_EXP_NAME:-{{ablation_name}}}"

summarize-private-aux-ablation:
    @PYTHONPATH=. {{python}} summarize_aux_ablation.py --runs-root "${TREEFORMER_ABLATION_RUNS_ROOT:-{{private_ablation_output}}/runs}" --output-csv "${TREEFORMER_ABLATION_CSV:-{{private_ablation_output}}/ablation_summary.csv}"

cfg-private-joint-virtual-root-aux:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. {{python}} train_hydra.py --cfg job train=joint_virtual_root_aux augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} checkpoint.pretrained_strict=false checkpoint.save_last=false DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE="${TREEFORMER_BATCH_SIZE:-2}" DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-640}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-0}" DATA.PERSISTENT_WORKERS=false DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY=full DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_joint_vr_aux_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" TRAIN.SMD_BACKEND="${TREEFORMER_SMD_BACKEND:-legacy}" TRAIN.SMD_N_POINTS="${TREEFORMER_SMD_N_POINTS:-500}" TRAIN.SMD_BLUR="${TREEFORMER_SMD_BLUR:-0.01}" runtime.compile.aux_head=false runtime.compile.aux_loss=false log.exp_name="${TREEFORMER_EXP_NAME:-private_joint_vr_aux_no_da}"

smoke-private-joint-virtual-root-aux:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=joint_virtual_root_aux augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} checkpoint.pretrained_strict=false checkpoint.save_last=false DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE="${TREEFORMER_BATCH_SIZE:-2}" DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-640}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-2}" DATA.PERSISTENT_WORKERS=true DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-none}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY=full DATA.TRAIN_LIMIT=4 DATA.VAL_LIMIT=2 TRAIN.SAVE_PATH={{private_joint_vr_aux_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-1}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" TRAIN.SMD_BACKEND="${TREEFORMER_SMD_BACKEND:-legacy}" TRAIN.SMD_N_POINTS="${TREEFORMER_SMD_N_POINTS:-500}" TRAIN.SMD_BLUR="${TREEFORMER_SMD_BLUR:-0.01}" runtime.compile.aux_head=false runtime.compile.aux_loss=false log.exp_name=private_joint_vr_aux_smoke_no_da

train-private-joint-virtual-root-aux:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} train_hydra.py train=joint_virtual_root_aux augmentation=disabled optimizer=muon_schedulefree ema=default ema.evaluate=false logging=tensorboard checkpoint.pretrained={{fork_grapevine_pretrained}} checkpoint.pretrained_key={{pretrained_key}} checkpoint.pretrained_strict=false checkpoint.save_last=false DATA.DATASET=treeformer-2D DATA.DATA_PATH="{{private_treeformer_data}}" DATA.BATCH_SIZE="${TREEFORMER_BATCH_SIZE:-2}" DATA.MAX_SIZE="${TREEFORMER_MAX_SIZE:-640}" DATA.NUM_WORKERS="${TREEFORMER_NUM_WORKERS:-0}" DATA.PERSISTENT_WORKERS=false DATA.PREFETCH_FACTOR=2 DATA.SEG_CACHE_MODE="${TREEFORMER_SEG_CACHE_MODE:-disk}" DATA.SEG_CACHE_ROOT="{{seg_cache_root}}" DATA.SEG_RESIZE_POLICY=full DATA.TRAIN_LIMIT=null DATA.VAL_LIMIT=null TRAIN.SAVE_PATH={{private_joint_vr_aux_output}} TRAIN.EPOCHS="${TREEFORMER_EPOCHS:-100}" TRAIN.LR="${TREEFORMER_LR:-1e-4}" TRAIN.LR_BACKBONE="${TREEFORMER_LR_BACKBONE:-3e-5}" TRAIN.SMD_BACKEND="${TREEFORMER_SMD_BACKEND:-legacy}" TRAIN.SMD_N_POINTS="${TREEFORMER_SMD_N_POINTS:-500}" TRAIN.SMD_BLUR="${TREEFORMER_SMD_BLUR:-0.01}" runtime.compile.aux_head=false runtime.compile.aux_loss=false log.exp_name="${TREEFORMER_EXP_NAME:-private_joint_vr_aux_no_da}"

tune-private-joint-virtual-root-aux:
    @test -n "{{private_treeformer_data}}" || (echo "Set TREEFORMER_PRIVATE_DATA to a legacy TreeFormer dataset root" >&2; exit 2)
    @PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{python}} tune_graph_optuna.py --trials "${TREEFORMER_OPTUNA_TRIALS:-20}" --epochs "${TREEFORMER_EPOCHS:-100}" --max-size "${TREEFORMER_MAX_SIZE:-640}" --batch-size "${TREEFORMER_BATCH_SIZE:-2}" --num-workers "${TREEFORMER_NUM_WORKERS:-0}" --train-limit "${TREEFORMER_TRAIN_LIMIT:-null}" --val-limit "${TREEFORMER_VAL_LIMIT:-null}" --output-root "{{private_optuna_joint_vr_aux_output}}" --assets-root "{{assets_root}}" --seg-cache-root "{{joint_optuna_cache_root}}" --seg-cache-mode "${TREEFORMER_SEG_CACHE_MODE:-disk}" --python "{{python}}" --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}"
