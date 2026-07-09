set dotenv-load := false

python := env_var_or_default("TREEFORMER_PYTHON", "../venv/TreeFormer/bin/python")
assets_root := env_var_or_default("TREEFORMER_ASSETS_ROOT", "../TreeFormer_assets")
private_treeformer_data := env_var_or_default("TREEFORMER_PRIVATE_DATA", "")
fork_grapevine_pretrained := env_var_or_default("TREEFORMER_PRETRAINED_CHECKPOINT", assets_root + "/pretrained_weights/fork_source_main/grapevein/checkpoint_ours.pkl")
private_pretrained_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT", assets_root + "/trained_weights_hydra_private_pretrained")
private_pretrained_aug_output := env_var_or_default("TREEFORMER_TRAIN_OUTPUT_AUG", assets_root + "/trained_weights_hydra_private_pretrained_aug")

hydra-cfg:
    PYTHONPATH=. {{python}} train_hydra.py --cfg job

test:
    PYTHONPATH=. {{python}} -m pytest -q

lint:
    PYTHONPATH=. {{python}} -m ruff check .

format:
    PYTHONPATH=. {{python}} -m ruff format .

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
