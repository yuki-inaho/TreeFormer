from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from treeformer_train.config import make_legacy_config


CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def test_hydra_default_config_composes_and_preserves_legacy_sections():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config")

    assert cfg.DATA.DATASET == "guyot-2D"
    assert cfg.MODEL.DECODER.OBJ_TOKEN == 256
    assert cfg.TRAIN.LR == 1e-4
    assert cfg.optimizer.name == "adamw_step"
    assert cfg.tensorboard.enabled is True
    assert cfg.checkpoint.metric_name == "val/smd"
    assert cfg.DATA.AUGMENTATION.enabled is False
    assert cfg.DATA.LEGACY_ROTATE is True

    legacy = make_legacy_config(cfg)
    assert legacy.DATA.DATASET == "guyot-2D"
    assert legacy.TRAIN.LOSSES == ["boxes", "class", "cards", "nodes", "edges"]


def test_hydra_muon_schedulefree_override_is_explicit():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["optimizer=muon_schedulefree", "ema=default", "train=dry_run"])

    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["optimizer"]["name"] == "muon_schedulefree"
    assert resolved["ema"]["enabled"] is True
    assert resolved["TRAIN"]["EPOCHS"] == 0
    assert "bias" in resolved["optimizer"]["aux_keywords"]


def test_hydra_regularized_augmentation_override_is_explicit():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["augmentation=regularized"])

    assert cfg.DATA.AUGMENTATION.enabled is True
    assert cfg.DATA.AUGMENTATION.photometric.backend == "albumentationsx"
    assert cfg.DATA.LEGACY_ROTATE is False

    legacy = make_legacy_config(cfg)
    assert legacy.DATA.AUGMENTATION.photometric.allow_fallback is True
