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
    assert cfg.DATA.LEGACY_ROTATE is False
    assert cfg.runtime.cuda.allow_tf32 is True
    assert cfg.runtime.cuda.cudnn_benchmark is True
    assert cfg.runtime.cuda.float32_matmul_precision == "high"
    assert cfg.runtime.compile.aux_head is False
    assert cfg.runtime.compile.aux_loss is False

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
    assert cfg.DATA.AUGMENTATION.photometric.backend == "opencv"
    assert cfg.DATA.LEGACY_ROTATE is False

    legacy = make_legacy_config(cfg)
    assert legacy.DATA.AUGMENTATION.photometric.allow_fallback is True


def test_hydra_albumentationsx_augmentation_is_opt_in():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["augmentation=regularized_albumentationsx"])

    assert cfg.DATA.AUGMENTATION.enabled is True
    assert cfg.DATA.AUGMENTATION.photometric.backend == "albumentationsx"
    assert cfg.DATA.AUGMENTATION.affine.enabled is True
    assert cfg.DATA.AUGMENTATION.elastic.enabled is True


def test_hydra_curriculum_augmentation_stages_compose():
    stages = {
        "photometric_opencv": (True, False, False),
        "regularized": (True, True, True),
        "geometry_mild": (True, True, True),
    }
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        for stage, expected in stages.items():
            cfg = compose(config_name="config", overrides=[f"augmentation={stage}"])
            photometric_enabled, affine_enabled, elastic_enabled = expected
            assert cfg.DATA.AUGMENTATION.photometric.enabled is photometric_enabled
            assert cfg.DATA.AUGMENTATION.affine.enabled is affine_enabled
            assert cfg.DATA.AUGMENTATION.elastic.enabled is elastic_enabled
            assert cfg.DATA.LEGACY_ROTATE is False


def test_hydra_aux_supervised_training_disables_graph_output():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=aux_supervised"])

    assert cfg.TRAIN.MODE == "aux_supervised"
    assert cfg.TRAIN.SKIP_GRAPH_OUTPUT is True
    assert cfg.TRAIN.LOSSES == []
    assert cfg.MODEL.GRAPH_OUTPUT_ENABLED is False
    assert cfg.MODEL.AUX_HEAD.ENABLED is True
    assert cfg.MODEL.AUX_HEAD.OUT_CHANNELS == 4
    assert cfg.TRAIN.W_AUX_DETAIL == 0.0
    assert cfg.checkpoint.metric_name == "val/aux_total_loss"
    assert cfg.checkpoint.pretrained_strict is False

    legacy = make_legacy_config(cfg)
    assert legacy.TRAIN.W_EDGE == 0.0
    assert legacy.MODEL.GRAPH_OUTPUT_ENABLED is False


def test_hydra_seg_supervised_training_uses_segmentation_only_losses():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=seg_supervised"])

    assert cfg.TRAIN.MODE == "aux_supervised"
    assert cfg.TRAIN.SKIP_GRAPH_OUTPUT is True
    assert cfg.TRAIN.W_AUX_SEG_BCE == 1.0
    assert cfg.TRAIN.W_AUX_SEG_DICE == 2.0
    assert cfg.TRAIN.W_AUX_SEG_FOCAL == 0.0
    assert cfg.TRAIN.AUX_SEG_POS_WEIGHT == "auto"
    assert cfg.TRAIN.AUX_SEG_POS_WEIGHT_MAX == 4.0
    assert cfg.TRAIN.W_AUX_DETAIL == 0.1
    assert cfg.TRAIN.W_AUX_DETAIL_BCE == 1.0
    assert cfg.TRAIN.W_AUX_DETAIL_DICE == 1.0
    assert cfg.TRAIN.AUX_DETAIL_SUPPORT_KERNEL_SIZE == 3
    assert cfg.TRAIN.W_AUX_HEATMAP == 0.0
    assert cfg.TRAIN.W_AUX_PAF == 0.0
    assert cfg.MODEL.AUX_HEAD.OUT_CHANNELS == 5
    assert cfg.DATA.SEGMENTATION_TARGET_SOURCE == "external_mask"
    assert cfg.DATA.LEGACY_ROTATE is False
    assert cfg.DATA.FAST_SEGMENTATION_LOADER is True
    assert cfg.DATA.SEG_CACHE_MODE == "none"
    assert cfg.DATA.AUX_TARGET_MODE == "seg_only"
    assert cfg.checkpoint.metric_name == "val/seg_soft_dice_score"
    assert cfg.checkpoint.mode == "max"

    legacy = make_legacy_config(cfg)
    assert legacy.TRAIN.W_AUX_HEATMAP == 0.0
    assert legacy.TRAIN.W_AUX_PAF == 0.0
    assert legacy.DATA.AUX_DETAIL_SCALES == [1, 2, 4]


def test_hydra_seg_only_training_is_pure_segmentation_objective():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=seg_only"])

    assert cfg.TRAIN.MODE == "aux_supervised"
    assert cfg.TRAIN.SKIP_GRAPH_OUTPUT is True
    assert cfg.TRAIN.W_AUX_SEG == 1.0
    assert cfg.TRAIN.W_AUX_DETAIL == 0.0
    assert cfg.TRAIN.W_AUX_HEATMAP == 0.0
    assert cfg.TRAIN.W_AUX_PAF == 0.0
    assert cfg.DATA.AUX_TARGET_MODE == "seg_only"
    assert cfg.DATA.FAST_SEGMENTATION_LOADER is True
    assert cfg.MODEL.GRAPH_OUTPUT_ENABLED is False
    assert cfg.MODEL.AUX_HEAD.OUT_CHANNELS == 5
    assert cfg.checkpoint.metric_name == "val/seg_soft_dice_score"


def test_hydra_seg_heatmap_training_adds_node_heatmap_objective():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=seg_heatmap"])

    assert cfg.TRAIN.MODE == "aux_supervised"
    assert cfg.TRAIN.SKIP_GRAPH_OUTPUT is True
    assert cfg.TRAIN.W_AUX_SEG == 1.0
    assert cfg.TRAIN.W_AUX_DETAIL == 0.0
    assert cfg.TRAIN.W_AUX_HEATMAP == 1.0
    assert cfg.TRAIN.W_AUX_HEATMAP_MSE == 1.0
    assert cfg.TRAIN.W_AUX_HEATMAP_FOCAL == 0.0
    assert cfg.TRAIN.W_AUX_HEATMAP_RIDGE == 0.0
    assert cfg.TRAIN.AUX_HEATMAP_MASK_SOURCE == "segmentation"
    assert cfg.TRAIN.AUX_HEATMAP_MASK_OUTSIDE_WEIGHT == 0.05
    assert cfg.TRAIN.W_AUX_PAF == 0.0
    assert cfg.DATA.AUX_TARGET_MODE == "seg_heatmap"
    assert cfg.DATA.AUX_HEATMAP_SIGMA == 3.0
    assert cfg.MODEL.GRAPH_OUTPUT_ENABLED is False
    assert cfg.MODEL.AUX_HEAD.OUT_CHANNELS == 5
    assert cfg.checkpoint.metric_name == "val/seg_soft_dice_score"


def test_hydra_seg_heatmap_paf_training_adds_edge_direction_objective():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=seg_heatmap_paf"])

    assert cfg.TRAIN.MODE == "aux_supervised"
    assert cfg.TRAIN.SKIP_GRAPH_OUTPUT is True
    assert cfg.TRAIN.W_AUX_SEG == 1.0
    assert cfg.TRAIN.W_AUX_HEATMAP == 1.0
    assert cfg.TRAIN.W_AUX_HEATMAP_MSE == 1.0
    assert cfg.TRAIN.W_AUX_HEATMAP_FOCAL == 0.0
    assert cfg.TRAIN.W_AUX_HEATMAP_RIDGE == 0.0
    assert cfg.TRAIN.AUX_HEATMAP_MASK_SOURCE == "segmentation"
    assert cfg.TRAIN.AUX_HEATMAP_MASK_OUTSIDE_WEIGHT == 0.05
    assert cfg.TRAIN.W_AUX_PAF == 0.25
    assert cfg.TRAIN.AUX_PAF_MASK_SOURCE == "paf_and_segmentation"
    assert cfg.DATA.AUX_TARGET_MODE == "seg_heatmap_paf"
    assert cfg.MODEL.GRAPH_OUTPUT_ENABLED is False
    assert cfg.MODEL.AUX_HEAD.OUT_CHANNELS == 5
    assert cfg.checkpoint.metric_name == "val/aux_total_loss"
    assert cfg.checkpoint.mode == "min"


def test_hydra_heatmap_ablation_group_overrides_loss_shape():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=seg_heatmap_paf", "+ablation=heatmap_focal_ridge"])

    assert cfg.TRAIN.W_AUX_HEATMAP_MSE == 0.25
    assert cfg.TRAIN.W_AUX_HEATMAP_FOCAL == 1.0
    assert cfg.TRAIN.W_AUX_HEATMAP_RIDGE == 0.1
    assert cfg.DATA.AUX_HEATMAP_SIGMA == 1.5
    assert cfg.TRAIN.W_AUX_SEG == 1.0


def test_hydra_heatmap_ablation_can_lower_segmentation_weight():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["train=seg_heatmap_paf", "+ablation=heatmap_focal_ridge_seg_low"],
        )

    assert cfg.TRAIN.W_AUX_SEG == 0.5
    assert cfg.TRAIN.W_AUX_HEATMAP_FOCAL == 1.0
    assert cfg.DATA.AUX_HEATMAP_SIGMA == 1.5


def test_hydra_virtual_root_config_composes():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=virtual_root"])

    assert cfg.TRAIN.MODE == "graph"
    assert cfg.TRAIN.VIRTUAL_ROOT is True
    assert cfg.TRAIN.POSTPROCESSOR_MODE == "vr-mst"
    assert "edges_virtual_root" in cfg.TRAIN.LOSSES
    assert "root" in cfg.TRAIN.LOSSES
    assert cfg.DATA.FOREST_METADATA is True
    assert cfg.DATA.STRICT_VIRTUAL_ROOT_METADATA is True
    assert cfg.MODEL.ROOT_HEAD.ENABLED is True

    legacy = make_legacy_config(cfg)
    assert legacy.TRAIN.VIRTUAL_ROOT is True
    assert legacy.MODEL.ROOT_HEAD.ENABLED is True


def test_hydra_joint_virtual_root_aux_config_enables_graph_and_all_aux_heads():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=["train=joint_virtual_root_aux"])

    assert cfg.TRAIN.MODE == "joint_graph_aux"
    assert cfg.TRAIN.SKIP_GRAPH_OUTPUT is False
    assert cfg.TRAIN.VIRTUAL_ROOT is True
    assert cfg.TRAIN.POSTPROCESSOR_MODE == "vr-mst"
    assert "edges_virtual_root" in cfg.TRAIN.LOSSES
    assert "root" in cfg.TRAIN.LOSSES
    assert cfg.TRAIN.W_JOINT_AUX > 0.0
    assert cfg.TRAIN.W_AUX_SEG > 0.0
    assert cfg.TRAIN.W_AUX_DETAIL > 0.0
    assert cfg.TRAIN.W_AUX_HEATMAP > 0.0
    assert cfg.TRAIN.W_AUX_PAF > 0.0
    assert cfg.TRAIN.AUX_HEATMAP_MASK_SOURCE == "segmentation"
    assert cfg.TRAIN.AUX_PAF_MASK_SOURCE == "paf_and_segmentation"
    assert cfg.DATA.FOREST_METADATA is True
    assert cfg.DATA.STRICT_VIRTUAL_ROOT_METADATA is True
    assert cfg.DATA.FAST_SEGMENTATION_LOADER is True
    assert cfg.DATA.AUX_TARGET_MODE == "seg_heatmap_paf"
    assert cfg.MODEL.GRAPH_OUTPUT_ENABLED is True
    assert cfg.MODEL.ROOT_HEAD.ENABLED is True
    assert cfg.MODEL.AUX_HEAD.ENABLED is True
    assert cfg.MODEL.AUX_HEAD.OUT_CHANNELS == 5
    assert cfg.checkpoint.metric_name == "val/smd"

    legacy = make_legacy_config(cfg)
    assert legacy.TRAIN.VIRTUAL_ROOT is True
    assert legacy.MODEL.GRAPH_OUTPUT_ENABLED is True
    assert legacy.MODEL.ROOT_HEAD.ENABLED is True
    assert legacy.MODEL.AUX_HEAD.ENABLED is True
