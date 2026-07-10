import torch

from treeformer_train.aux_training import (
    AuxLossWeights,
    build_aux_loss_computer,
    build_aux_loss_weights,
    compute_aux_eval_metrics,
    compute_aux_losses,
)


class Config:
    W_AUX_SEG = 2.0
    W_AUX_SEG_DICE = 1.5
    AUX_SEG_POS_WEIGHT = "auto"
    W_AUX_DETAIL = 0.1
    W_AUX_DETAIL_DICE = 0.5
    W_AUX_HEATMAP = 3.0
    AUX_HEATMAP_MASK_SOURCE = "segmentation"
    AUX_HEATMAP_MASK_OUTSIDE_WEIGHT = 0.05
    W_AUX_HEATMAP_MSE = 0.25
    W_AUX_HEATMAP_FOCAL = 1.0
    W_AUX_HEATMAP_RIDGE = 0.1
    W_AUX_PAF = 0.5
    AUX_PAF_MASK_SOURCE = "paf_and_segmentation"


def _targets(batch_size=2, height=8, width=10):
    segmentation = torch.zeros(batch_size, 1, height, width)
    segmentation[:, :, 2:6, 3:7] = 1.0
    heatmap = torch.zeros(batch_size, 1, height, width)
    heatmap[:, :, 4, 5] = 1.0
    paf = torch.zeros(batch_size, 2, height, width)
    paf[:, 0, 2:6, 3:7] = 1.0
    paf_mask = segmentation.bool()
    return {
        "segmentation": segmentation,
        "heatmap": heatmap,
        "paf": paf,
        "paf_mask": paf_mask,
    }


def test_build_aux_loss_weights_reads_legacy_train_config():
    weights = build_aux_loss_weights(Config())

    assert weights.segmentation == 2.0
    assert weights.segmentation_bce == 1.0
    assert weights.segmentation_dice == 1.5
    assert weights.segmentation_pos_weight == "auto"
    assert weights.detail == 0.1
    assert weights.detail_dice == 0.5
    assert weights.detail_scales == (1, 2, 4)
    assert weights.heatmap == 3.0
    assert weights.heatmap_mask_source == "segmentation"
    assert weights.heatmap_mask_outside_weight == 0.05
    assert weights.heatmap_mse == 0.25
    assert weights.heatmap_focal == 1.0
    assert weights.heatmap_ridge == 0.1
    assert weights.paf == 0.5
    assert weights.paf_mask_source == "paf_and_segmentation"


def test_compute_aux_losses_backpropagates_with_resized_maps():
    output = {"aux_maps": torch.randn(2, 4, 4, 5, requires_grad=True)}
    losses = compute_aux_losses(output, _targets(), AuxLossWeights())

    assert set(losses) == {
        "total",
        "seg_total",
        "seg_bce",
        "seg_dice",
        "seg_focal",
        "detail_total",
        "detail_bce",
        "detail_dice",
        "heatmap_total",
        "heatmap_mse",
        "heatmap_focal",
        "heatmap_ridge",
        "paf_l1",
    }
    assert losses["total"].ndim == 0

    losses["total"].backward()

    assert output["aux_maps"].grad is not None
    assert torch.isfinite(output["aux_maps"].grad).all()


def test_aux_loss_computer_matches_direct_loss_terms():
    output = {"aux_maps": torch.randn(2, 5, 8, 10, requires_grad=True)}
    targets = _targets()
    weights = AuxLossWeights(segmentation_dice=1.0, detail=0.1, heatmap=0.0, paf=0.0)

    direct = compute_aux_losses(output, targets, weights)
    computer = build_aux_loss_computer(weights)
    computed = computer(output, targets)

    assert direct.keys() == computed.keys()
    for key in direct:
        assert torch.allclose(direct[key], computed[key])


def test_heatmap_loss_can_be_weighted_by_segmentation_mask():
    targets = _targets(batch_size=1)
    aux_maps = torch.full((1, 4, 8, 10), -20.0)
    aux_maps[:, 1, :, :3] = 20.0
    unmasked = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(segmentation=0.0, heatmap=1.0, paf=0.0),
    )
    masked = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(
            segmentation=0.0,
            heatmap=1.0,
            heatmap_mask_source="segmentation",
            heatmap_mask_outside_weight=0.0,
            paf=0.0,
        ),
    )

    assert masked["heatmap_mse"] < unmasked["heatmap_mse"]


def test_heatmap_ablation_losses_are_differentiable():
    targets = _targets(batch_size=1)
    aux_maps = torch.randn(1, 4, 8, 10, requires_grad=True)
    losses = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(
            segmentation=0.0,
            heatmap=1.0,
            heatmap_mse=0.25,
            heatmap_focal=1.0,
            heatmap_ridge=0.1,
            heatmap_mask_source="segmentation",
            heatmap_mask_outside_weight=0.05,
            paf=0.0,
        ),
    )

    expected = 0.25 * losses["heatmap_mse"] + losses["heatmap_focal"] + 0.1 * losses["heatmap_ridge"]
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert aux_maps.grad is not None
    assert torch.isfinite(aux_maps.grad).all()


def test_paf_loss_can_intersect_paf_and_segmentation_masks():
    targets = _targets(batch_size=1)
    targets["paf_mask"] = torch.ones_like(targets["segmentation"], dtype=torch.bool)
    aux_maps = torch.zeros(1, 4, 8, 10)
    aux_maps[:, 2:4, :, :3] = 20.0
    aux_maps[:, 2, 2:6, 3:7] = 20.0
    paf_only = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(segmentation=0.0, heatmap=0.0, paf=1.0),
    )
    segmented = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(
            segmentation=0.0,
            heatmap=0.0,
            paf=1.0,
            paf_mask_source="paf_and_segmentation",
        ),
    )

    assert segmented["paf_l1"] < paf_only["paf_l1"]


def test_compute_aux_eval_metrics_reports_validation_signals():
    output = {"aux_maps": torch.zeros(2, 4, 8, 10)}
    metrics = compute_aux_eval_metrics(output, _targets(), AuxLossWeights())

    assert "total" in metrics
    assert "seg_iou" in metrics
    assert "seg_dice_score" in metrics
    assert "seg_soft_dice_score" in metrics
    assert "seg_precision" in metrics
    assert "seg_recall" in metrics
    assert "detail_iou" in metrics
    assert "detail_soft_dice_score" in metrics
    assert "heatmap_mae" in metrics
    assert "masked_heatmap_mae" in metrics
    assert "heatmap_peak_mean" in metrics
    assert "heatmap_nonpeak_foreground_mean" in metrics
    assert "heatmap_peak_contrast" in metrics
    assert "paf_masked_l1" in metrics
    assert 0.0 <= metrics["seg_iou"].item() <= 1.0
    assert 0.0 <= metrics["seg_soft_dice_score"].item() <= 1.0


def test_segmentation_only_weights_exclude_heatmap_and_paf_from_total():
    targets = _targets(batch_size=1)
    aux_maps = torch.randn(1, 4, 8, 10, requires_grad=True)
    weights = AuxLossWeights(
        segmentation=1.0,
        segmentation_bce=1.0,
        segmentation_dice=1.0,
        segmentation_focal=0.5,
        segmentation_pos_weight="auto",
        heatmap=0.0,
        paf=0.0,
    )

    losses = compute_aux_losses({"aux_maps": aux_maps}, targets, weights)

    expected = losses["seg_bce"] + losses["seg_dice"] + 0.5 * losses["seg_focal"]
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert aux_maps.grad is not None


def test_detail_boundary_loss_is_optional_and_uses_fifth_channel():
    targets = _targets(batch_size=1)
    aux_maps = torch.randn(1, 5, 8, 10, requires_grad=True)
    weights = AuxLossWeights(
        segmentation=1.0,
        segmentation_bce=1.0,
        detail=0.1,
        detail_bce=1.0,
        detail_dice=1.0,
        heatmap=0.0,
        paf=0.0,
    )

    losses = compute_aux_losses({"aux_maps": aux_maps}, targets, weights)

    expected = losses["seg_bce"] + 0.1 * (losses["detail_bce"] + losses["detail_dice"])
    assert torch.allclose(losses["total"], expected)
    assert losses["detail_total"].item() > 0.0
    losses["total"].backward()
    assert aux_maps.grad is not None


def test_detail_boundary_loss_requires_fifth_channel_when_enabled():
    targets = _targets(batch_size=1)
    aux_maps = torch.randn(1, 4, 8, 10, requires_grad=True)

    try:
        compute_aux_losses({"aux_maps": aux_maps}, targets, AuxLossWeights(detail=0.1))
    except ValueError as exc:
        assert "OUT_CHANNELS>=5" in str(exc)
    else:
        raise AssertionError("expected detail loss to require a fifth aux channel")


def test_segmentation_loss_rejects_unnormalized_mask_values():
    targets = _targets(batch_size=1)
    targets["segmentation"] = targets["segmentation"] * 255.0
    aux_maps = torch.randn(1, 4, 8, 10, requires_grad=True)

    try:
        compute_aux_losses({"aux_maps": aux_maps}, targets, AuxLossWeights())
    except ValueError as exc:
        assert "normalized to [0, 1]" in str(exc)
    else:
        raise AssertionError("expected unnormalized 0/255 segmentation target to be rejected")


def test_segmentation_loss_treats_mask_as_binary_foreground_target():
    targets = _targets(batch_size=1)
    targets["segmentation"] = targets["segmentation"].clamp(1e-7, 0.9999999)
    aux_maps = torch.randn(1, 4, 8, 10, requires_grad=True)

    losses = compute_aux_losses({"aux_maps": aux_maps}, targets, AuxLossWeights(segmentation_dice=1.0))

    assert torch.isfinite(losses["total"])
