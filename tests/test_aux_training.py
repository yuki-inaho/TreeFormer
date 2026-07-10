import torch

from treeformer_train.aux_map_targets import make_node_heatmap
from treeformer_train.aux_training import (
    AuxLossWeights,
    _extract_target_peak_indices,
    _local_softargmax_losses,
    _peakness_margin_loss,
    build_aux_loss_computer,
    build_aux_loss_weights,
    centernet_heatmap_focal_loss_with_logits,
    compute_aux_eval_metrics,
    compute_aux_losses,
)
from treeformer_train.heatmap_offsets import decode_native_heatmap_peaks, make_native_offset_targets


class Config:
    W_AUX_SEG = 2.0
    W_AUX_SEG_DICE = 1.5
    AUX_SEG_POS_WEIGHT = "auto"
    W_AUX_DETAIL = 0.1
    W_AUX_DETAIL_DICE = 0.5
    W_AUX_HEATMAP = 3.0
    AUX_HEATMAP_MASK_SOURCE = "segmentation"
    AUX_HEATMAP_MASK_OUTSIDE_WEIGHT = 0.0
    W_AUX_HEATMAP_MSE = 0.25
    W_AUX_HEATMAP_FOCAL = 1.0
    W_AUX_HEATMAP_RIDGE = 0.1
    W_AUX_PAF = 0.5
    AUX_PAF_MASK_SOURCE = "paf_and_segmentation"
    AUX_DIRECTION_ENCODING = "double_angle"


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
    assert weights.heatmap_mask_outside_weight == 0.0
    assert weights.heatmap_mse == 0.25
    assert weights.heatmap_focal == 1.0
    assert weights.heatmap_ridge == 0.1
    assert weights.paf == 0.5
    assert weights.paf_mask_source == "paf_and_segmentation"
    assert weights.direction_encoding == "double_angle"


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
        "heatmap_coord",
        "heatmap_coord_var",
        "heatmap_peak",
        "heatmap_offset",
        "paf_total",
        "paf_l1",
        "paf_angular",
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
            heatmap_mask_outside_weight=0.0,
            paf=0.0,
        ),
    )

    expected = 0.25 * losses["heatmap_mse"] + losses["heatmap_focal"] + 0.1 * losses["heatmap_ridge"]
    assert torch.allclose(losses["total"], expected)
    losses["total"].backward()
    assert aux_maps.grad is not None
    assert torch.isfinite(aux_maps.grad).all()


def test_heatmap_mask_excludes_outside_pixels_from_loss_gradient():
    targets = _targets(batch_size=1)
    aux_maps = torch.zeros(1, 4, 8, 10, requires_grad=True)
    losses = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(
            segmentation=0.0,
            heatmap=1.0,
            heatmap_mask_source="segmentation",
            heatmap_mask_outside_weight=0.0,
            heatmap_mse=1.0,
            heatmap_focal=0.0,
            heatmap_ridge=0.0,
            paf=0.0,
        ),
    )
    losses["total"].backward()
    outside = targets["segmentation"] == 0.0
    assert torch.count_nonzero(aux_maps.grad[:, 1:2][outside]) == 0


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
    assert "direction_angular_error_deg" in metrics
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


def test_aux_loss_does_not_recompute_detail_target_when_provided():
    from unittest.mock import patch

    from treeformer_train.detail_targets import make_stdc_detail_boundary_target

    targets = _targets(batch_size=1)
    aux_maps = torch.randn(1, 5, 8, 10, requires_grad=True)
    weights = AuxLossWeights(detail=0.1, detail_bce=1.0, detail_dice=1.0, heatmap=0.0, paf=0.0)

    seg_target = (targets["segmentation"] > 0.5).to(dtype=targets["segmentation"].dtype)
    precomputed_detail_target = make_stdc_detail_boundary_target(
        seg_target,
        threshold=weights.detail_threshold,
        scales=weights.detail_scales,
        support_kernel_size=weights.detail_support_kernel_size,
    )

    with patch("treeformer_train.aux_training.make_stdc_detail_boundary_target") as mock_make_detail:
        losses = compute_aux_losses(
            {"aux_maps": aux_maps},
            targets,
            weights,
            detail_target=precomputed_detail_target,
        )

    mock_make_detail.assert_not_called()
    assert losses["detail_total"].item() != 0.0


def test_compute_aux_eval_metrics_computes_detail_target_only_once():
    from unittest.mock import patch

    from treeformer_train.detail_targets import make_stdc_detail_boundary_target

    aux_maps = torch.randn(2, 5, 8, 10)
    targets = _targets()
    weights = AuxLossWeights(detail=0.1, heatmap=0.0, paf=0.0)

    with patch(
        "treeformer_train.aux_training.make_stdc_detail_boundary_target",
        wraps=make_stdc_detail_boundary_target,
    ) as mock_make_detail:
        metrics = compute_aux_eval_metrics({"aux_maps": aux_maps}, targets, weights)

    assert mock_make_detail.call_count == 1
    assert "detail_iou" in metrics


def test_aux_loss_requires_detail_target_when_fifth_channel_active_and_not_provided():
    targets = _targets(batch_size=1)
    aux_maps = torch.randn(1, 5, 8, 10, requires_grad=True)
    weights = AuxLossWeights(detail=0.1, detail_bce=1.0, detail_dice=1.0, heatmap=0.0, paf=0.0)
    seg_target = (targets["segmentation"] > 0.5).to(dtype=targets["segmentation"].dtype)
    heatmap_target = targets["heatmap"]
    paf_target = targets["paf"]
    paf_mask = targets["paf_mask"].to(dtype=torch.float32)

    from treeformer_train.aux_training import _compute_aux_loss_terms

    try:
        _compute_aux_loss_terms(
            aux_maps,
            aux_maps[:, 1:2],
            seg_target,
            heatmap_target,
            paf_target,
            paf_mask,
            weights,
            detail_target=None,
        )
    except ValueError as exc:
        assert "detail_target" in str(exc)
    else:
        raise AssertionError("expected _compute_aux_loss_terms to reject a missing detail_target on a 5ch head")


def test_native_heatmap_logits_are_used_for_stride_target_supervision():
    targets = _targets(batch_size=1, height=8, width=12)
    targets["heatmap"] = torch.zeros(1, 1, 2, 3)
    targets["heatmap"][:, :, 1, 1] = 1.0
    aux_maps = torch.zeros(1, 4, 8, 12, requires_grad=True)
    native_logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
    losses = compute_aux_losses(
        {"aux_maps": aux_maps, "aux_heatmap_native": native_logits},
        targets,
        AuxLossWeights(segmentation=0.0, heatmap=1.0, paf=0.0),
    )

    losses["total"].backward()

    assert native_logits.grad is not None
    assert torch.count_nonzero(native_logits.grad) > 0
    assert aux_maps.grad is not None
    assert torch.count_nonzero(aux_maps.grad) == 0

    metrics = compute_aux_eval_metrics(
        {"aux_maps": aux_maps.detach(), "aux_heatmap_native": native_logits.detach()},
        targets,
        AuxLossWeights(segmentation=0.0, heatmap=1.0, paf=0.0),
    )
    assert torch.isfinite(metrics["heatmap_peak_contrast"])
    assert torch.isfinite(metrics["heatmap_node_recall"])


def test_native_heatmap_target_rejects_legacy_full_resolution_head():
    targets = _targets(batch_size=1, height=8, width=12)
    targets["heatmap"] = torch.zeros(1, 1, 2, 3)
    aux_maps = torch.zeros(1, 4, 8, 12)

    try:
        compute_aux_losses({"aux_maps": aux_maps}, targets, AuxLossWeights())
    except KeyError as exc:
        assert "aux_heatmap_native" in str(exc)
    else:
        raise AssertionError("expected native target supervision to require native heatmap logits")


def test_native_offset_targets_and_nms_decode_preserve_subcell_coordinates():
    offsets, valid = make_native_offset_targets(
        [torch.tensor([[0.6, 0.5]], dtype=torch.float32)],
        target_size=(4, 6),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert valid.sum().item() == 1
    assert torch.allclose(offsets[0, :, 2, 3], torch.tensor([0.0, -0.5]))

    heatmap_logits = torch.full((1, 1, 4, 6), -12.0)
    heatmap_logits[0, 0, 2, 3] = 12.0
    offset_logits = torch.zeros((1, 2, 4, 6))
    decoded = decode_native_heatmap_peaks(heatmap_logits, offset_logits, threshold=0.25)

    assert len(decoded) == 1
    assert decoded[0].shape == (1, 3)
    assert torch.allclose(decoded[0][0, :2], torch.tensor([3.0, 2.0]))

    rejected = decode_native_heatmap_peaks(
        heatmap_logits,
        offset_logits,
        threshold=0.25,
        valid_mask=torch.zeros_like(heatmap_logits, dtype=torch.bool),
    )
    assert rejected[0].shape == (0, 3)


def test_native_offset_loss_uses_node_coordinates_without_changing_collate_targets():
    targets = _targets(batch_size=1, height=8, width=12)
    targets["heatmap"] = torch.zeros(1, 1, 2, 3)
    targets["nodes"] = [torch.tensor([[0.5, 0.5]], dtype=torch.float32)]
    aux_maps = torch.zeros(1, 4, 8, 12, requires_grad=True)
    native_logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
    offset_logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
    weights = AuxLossWeights(
        segmentation=0.0,
        heatmap=1.0,
        heatmap_mse=0.0,
        heatmap_focal=0.0,
        heatmap_ridge=0.0,
        heatmap_offset=1.0,
        paf=0.0,
    )

    losses = compute_aux_losses(
        {
            "aux_maps": aux_maps,
            "aux_heatmap_native": native_logits,
            "aux_heatmap_offset_native": offset_logits,
        },
        targets,
        weights,
    )
    losses["total"].backward()

    assert losses["heatmap_offset"].item() > 0.0
    assert offset_logits.grad is not None
    assert torch.count_nonzero(offset_logits.grad) > 0


def test_compute_aux_eval_metrics_returns_only_scalar_tensors():
    # _MetricAverager.update (aux_training.py) calls float(value.detach().item())
    # on every value in the metrics dict. A non-scalar tensor sneaking into the
    # dict (e.g. a raw detail_target carried through for reuse) crashes
    # epoch_val_aux / epoch_val_aux_from_joint_loader at runtime even though this
    # function's own unit tests pass, because no existing test exercises the
    # averaging step on a 5ch aux head.
    aux_maps = torch.randn(2, 5, 8, 10)
    targets = _targets()
    weights = AuxLossWeights(detail=0.1, heatmap=0.0, paf=0.0)

    metrics = compute_aux_eval_metrics({"aux_maps": aux_maps}, targets, weights)

    for key, value in metrics.items():
        assert value.ndim == 0, f"metrics['{key}'] must be a scalar tensor, got shape {tuple(value.shape)}"


def test_compute_aux_eval_metrics_output_survives_metric_averaging():
    from treeformer_train.aux_training import _MetricAverager

    aux_maps = torch.randn(2, 5, 8, 10)
    targets = _targets()
    weights = AuxLossWeights(detail=0.1, heatmap=0.0, paf=0.0)

    metrics = compute_aux_eval_metrics({"aux_maps": aux_maps}, targets, weights)

    averager = _MetricAverager()
    averager.update(metrics, targets["segmentation"].shape[0])


def test_compute_aux_losses_returns_only_scalars_for_training_metric_averaging():
    from treeformer_train.aux_training import _MetricAverager

    targets = _targets()
    aux_maps = torch.randn(2, 5, 8, 10, requires_grad=True)
    losses = compute_aux_losses(
        {"aux_maps": aux_maps},
        targets,
        AuxLossWeights(detail=0.1, heatmap=0.0, paf=0.0),
    )

    assert all(value.ndim == 0 for value in losses.values())
    _MetricAverager().update(losses, targets["segmentation"].shape[0])
    losses["total"].backward()
    assert aux_maps.grad is not None


class PeakCoordConfig:
    AUX_HEATMAP_FOCAL_POS_SOURCE = "target_peaks"
    W_AUX_HEATMAP_COORD = 0.5
    AUX_HEATMAP_COORD_WINDOW_RADIUS = 4
    AUX_HEATMAP_COORD_TEMPERATURE = 2.0
    AUX_HEATMAP_COORD_HUBER_DELTA = 0.5
    W_AUX_HEATMAP_COORD_VAR = 0.3
    W_AUX_HEATMAP_PEAK = 0.2
    AUX_HEATMAP_PEAK_CENTER_RADIUS = 2
    AUX_HEATMAP_PEAK_ANNULUS_INNER = 4
    AUX_HEATMAP_PEAK_ANNULUS_OUTER = 8
    AUX_HEATMAP_PEAK_MARGIN = 2.0
    AUX_HEATMAP_PEAK_TEMPERATURE = 0.5
    AUX_HEATMAP_PEAK_MIN_TARGET = 0.6
    AUX_HEATMAP_EVAL_PEAK_THRESHOLD = 0.7
    AUX_HEATMAP_EVAL_MATCH_RADIUS = 8.0


def test_build_aux_loss_weights_reads_heatmap_peak_and_coord_keys():
    weights = build_aux_loss_weights(PeakCoordConfig())

    assert weights.heatmap_focal_pos_source == "target_peaks"
    assert weights.heatmap_coord == 0.5
    assert weights.heatmap_coord_window_radius == 4
    assert weights.heatmap_coord_temperature == 2.0
    assert weights.heatmap_coord_huber_delta == 0.5
    assert weights.heatmap_coord_var == 0.3
    assert weights.heatmap_peak == 0.2
    assert weights.heatmap_peak_center_radius == 2
    assert weights.heatmap_peak_annulus_inner == 4
    assert weights.heatmap_peak_annulus_outer == 8
    assert weights.heatmap_peak_margin == 2.0
    assert weights.heatmap_peak_temperature == 0.5
    assert weights.heatmap_peak_min_target == 0.6
    assert weights.heatmap_eval_peak_threshold == 0.7
    assert weights.heatmap_eval_match_radius == 8.0


def test_extract_target_peaks_recovers_offgrid_gaussian_centers():
    # Off-grid by 0.45px in both axes: close enough to a pixel center to stay a
    # clean local maximum, far enough that the rendered Gaussian peak value is
    # measurably below 1.0 -- this is the gap the threshold-based focal positive
    # rule misses and _extract_target_peak_indices is meant to recover.
    nodes = torch.tensor(
        [
            [6.45 / 31.0, 9.45 / 31.0],
            [22.45 / 31.0, 25.45 / 31.0],
        ]
    )
    heatmap = make_node_heatmap(nodes, (32, 32), sigma=3.0, cutoff=0.01)
    target = heatmap.unsqueeze(0).unsqueeze(0)

    assert target.max().item() < 0.99

    peaks = _extract_target_peak_indices(target, min_target=0.5)

    assert peaks.shape == (2, 3)
    assert peaks[0].tolist() == [0, 9, 6]
    assert peaks[1].tolist() == [0, 25, 22]


def test_focal_target_peaks_source_supervises_offgrid_nodes():
    nodes = torch.tensor([[6.45 / 11.0, 7.45 / 11.0]])
    heatmap = make_node_heatmap(nodes, (12, 12), sigma=3.0, cutoff=0.01)
    target = heatmap.unsqueeze(0).unsqueeze(0)
    assert target.max().item() < 0.99

    base = torch.zeros_like(target)
    confident = base.clone()
    confident[0, 0, 7, 6] = 10.0
    unconfident = base.clone()
    unconfident[0, 0, 7, 6] = -10.0

    # Without a positive_mask, target < pos_threshold everywhere so both pixels
    # fall on the negative branch, whose weight (1 - target)^beta vanishes near
    # the peak -- confident vs unconfident predictions there barely move the loss.
    loss_confident_no_mask = centernet_heatmap_focal_loss_with_logits(
        confident, target, pos_threshold=0.99
    )
    loss_unconfident_no_mask = centernet_heatmap_focal_loss_with_logits(
        unconfident, target, pos_threshold=0.99
    )
    assert abs(loss_confident_no_mask.item() - loss_unconfident_no_mask.item()) < 1e-4

    peaks = _extract_target_peak_indices(target, min_target=0.5)
    positive_mask = torch.zeros_like(target, dtype=torch.bool)
    positive_mask[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]] = True

    loss_confident_with_mask = centernet_heatmap_focal_loss_with_logits(
        confident, target, positive_mask=positive_mask
    )
    loss_unconfident_with_mask = centernet_heatmap_focal_loss_with_logits(
        unconfident, target, positive_mask=positive_mask
    )
    assert loss_confident_with_mask.item() < loss_unconfident_with_mask.item()


def test_focal_loss_normalizes_by_positive_pixel_count():
    import math

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1, 1] = 1.0
    logits = torch.zeros(1, 1, 4, 4)

    loss = centernet_heatmap_focal_loss_with_logits(logits, target, alpha=2.0, beta=4.0, pos_threshold=0.99)

    per_pixel_term = -(0.5**2) * math.log(0.5)
    expected = 16 * per_pixel_term / 1.0  # normalized by positive-pixel count (1), not all 16 pixels
    assert abs(loss.item() - expected) < 1e-5


def test_local_softargmax_loss_is_small_for_centered_peak_and_grows_when_shifted():
    peaks = torch.tensor([[0, 10, 10]], dtype=torch.long)

    # A gap of 12 (peak=6, baseline=-6) at temperature=1.0 concentrates the softmax
    # heavily on the spike (pi ~= 0.999) without saturating to exact 1.0 in float32,
    # so the gradient through softmax stays representable.
    centered_logits = torch.full((1, 1, 20, 20), -6.0)
    centered_logits[0, 0, 10, 10] = 6.0
    centered_logits.requires_grad_(True)
    coord_centered, _ = _local_softargmax_losses(
        centered_logits, peaks, window_radius=6, temperature=1.0, huber_delta=1.0
    )
    assert coord_centered.item() < 1e-3

    coord_centered.backward()
    assert centered_logits.grad is not None
    assert torch.count_nonzero(centered_logits.grad) > 0

    shifted_logits = torch.full((1, 1, 20, 20), -6.0)
    shifted_logits[0, 0, 13, 10] = 6.0
    coord_shifted, _ = _local_softargmax_losses(
        shifted_logits, peaks, window_radius=6, temperature=1.0, huber_delta=1.0
    )
    assert coord_shifted.item() > 1.0
    assert coord_shifted.item() > coord_centered.item()


def test_peakness_margin_loss_penalizes_flat_ridge_through_node():
    peaks = torch.tensor([[0, 10, 10]], dtype=torch.long)

    isolated_logits = torch.full((1, 1, 20, 20), -10.0)
    isolated_logits[0, 0, 10, 10] = 20.0
    isolated_loss = _peakness_margin_loss(
        isolated_logits,
        peaks,
        center_radius=1,
        annulus_inner=3,
        annulus_outer=6,
        margin=1.0,
        temperature=1.0,
    )

    ridge_logits = torch.full((1, 1, 20, 20), -10.0)
    ridge_logits[0, 0, 10, :] = 20.0
    ridge_loss = _peakness_margin_loss(
        ridge_logits,
        peaks,
        center_radius=1,
        annulus_inner=3,
        annulus_outer=6,
        margin=1.0,
        temperature=1.0,
    )

    assert isolated_loss.item() < 0.05
    assert ridge_loss.item() > 0.5
    assert ridge_loss.item() > isolated_loss.item()


def test_heatmap_peakness_ignores_pixels_outside_segmentation_mask():
    targets = _targets(batch_size=1)
    weights = AuxLossWeights(
        segmentation=0.0,
        heatmap=1.0,
        heatmap_mse=0.0,
        heatmap_focal=0.0,
        heatmap_ridge=0.0,
        heatmap_peak=1.0,
        heatmap_mask_source="segmentation",
        heatmap_mask_outside_weight=0.0,
        paf=0.0,
    )
    inside_logits = torch.zeros(1, 4, 8, 10)
    outside_logits = inside_logits.clone()
    outside = targets["segmentation"] == 0.0
    outside_logits[:, 1:2][outside] = 20.0

    inside_loss = compute_aux_losses({"aux_maps": inside_logits}, targets, weights)
    outside_loss = compute_aux_losses({"aux_maps": outside_logits}, targets, weights)

    assert torch.allclose(inside_loss["heatmap_peak"], outside_loss["heatmap_peak"])


def test_heatmap_peak_metrics_ignore_background_predictions():
    targets = _targets(batch_size=1)
    weights = AuxLossWeights(
        segmentation=0.0,
        heatmap=0.0,
        paf=0.0,
        heatmap_mask_source="segmentation",
        heatmap_mask_outside_weight=0.0,
        heatmap_peak_min_target=0.5,
        heatmap_eval_peak_threshold=0.5,
    )
    output = {"aux_maps": torch.full((1, 4, 8, 10), -20.0)}
    output["aux_maps"][:, 1, 4, 5] = 20.0
    output["aux_maps"][:, 1, 0, 0] = 20.0

    metrics = compute_aux_eval_metrics(output, targets, weights)

    assert metrics["heatmap_node_precision"] == 1.0
    assert metrics["heatmap_background_peaks_per_image"] == 0.0


def test_peakness_geometry_validation_rejects_bad_radii():
    logits = torch.zeros(1, 1, 10, 10)
    peaks = torch.zeros((0, 3), dtype=torch.long)

    try:
        _peakness_margin_loss(
            logits, peaks, center_radius=3, annulus_inner=3, annulus_outer=6, margin=1.0, temperature=1.0
        )
    except ValueError as exc:
        assert "AUX_HEATMAP_PEAK_CENTER_RADIUS" in str(exc)
    else:
        raise AssertionError("expected annulus_inner <= center_radius to raise ValueError")

    try:
        _peakness_margin_loss(
            logits, peaks, center_radius=1, annulus_inner=5, annulus_outer=3, margin=1.0, temperature=1.0
        )
    except ValueError as exc:
        assert "AUX_HEATMAP_PEAK_CENTER_RADIUS" in str(exc)
    else:
        raise AssertionError("expected annulus_outer < annulus_inner to raise ValueError")


def test_paf_angular_loss_is_wired_into_paf_total():
    targets = _targets(batch_size=1)
    base_aux_maps = torch.zeros(1, 4, 8, 10)
    weights = AuxLossWeights(segmentation=0.0, heatmap=0.0, paf=1.0, paf_l1=1.0, paf_angular=1.0)

    orthogonal = base_aux_maps.clone()
    orthogonal[:, 3, 2:6, 3:7] = 20.0  # predicted direction ~(0, 1) vs target (1, 0)
    losses_orthogonal = compute_aux_losses({"aux_maps": orthogonal}, targets, weights)
    assert losses_orthogonal["paf_angular"].item() > 0.0
    assert losses_orthogonal["paf_total"].item() > weights.paf_l1 * losses_orthogonal["paf_l1"].item()

    aligned = base_aux_maps.clone()
    aligned[:, 2, 2:6, 3:7] = 20.0  # predicted direction ~(1, 0), aligned with target
    losses_aligned = compute_aux_losses({"aux_maps": aligned}, targets, weights)
    assert losses_aligned["paf_angular"].item() < 1e-3


def test_compute_aux_eval_metrics_reports_peak_detection_quality():
    nodes = torch.tensor(
        [
            [8.3 / 31.0, 9.3 / 31.0],
            [24.6 / 31.0, 22.4 / 31.0],
        ]
    )
    heatmap = make_node_heatmap(nodes, (32, 32), sigma=3.0, cutoff=0.01)
    heatmap_target = heatmap.unsqueeze(0).unsqueeze(0)
    logits = torch.logit(heatmap_target.clamp(1e-4, 1.0 - 1e-4))

    aux_maps = torch.zeros(1, 4, 32, 32)
    aux_maps[:, 1:2] = logits
    targets = {
        "segmentation": torch.zeros(1, 1, 32, 32),
        "heatmap": heatmap_target,
        "paf": torch.zeros(1, 2, 32, 32),
        "paf_mask": torch.zeros(1, 1, 32, 32, dtype=torch.bool),
    }
    weights = AuxLossWeights(heatmap_peak_min_target=0.5, heatmap_eval_peak_threshold=0.5, heatmap_eval_match_radius=6.0)

    metrics = compute_aux_eval_metrics({"aux_maps": aux_maps}, targets, weights)

    assert abs(metrics["heatmap_node_recall"].item() - 1.0) < 1e-5
    assert abs(metrics["heatmap_node_precision"].item() - 1.0) < 1e-5
    assert abs(metrics["heatmap_duplicate_peak_rate"].item() - 0.0) < 1e-5
    assert abs(metrics["heatmap_background_peaks_per_image"].item() - 0.0) < 1e-5

    spurious_aux_maps = aux_maps.clone()
    spurious_aux_maps[:, 1, 28, 2] = 10.0
    metrics_with_spurious = compute_aux_eval_metrics({"aux_maps": spurious_aux_maps}, targets, weights)

    assert metrics_with_spurious["heatmap_node_recall"].item() > 0.99
    assert metrics_with_spurious["heatmap_node_precision"].item() < 1.0
    assert abs(metrics_with_spurious["heatmap_background_peaks_per_image"].item() - 1.0) < 1e-5


def test_disabled_new_heatmap_terms_leave_total_unchanged():
    targets = _targets(batch_size=2)
    aux_maps = torch.randn(2, 4, 8, 10, requires_grad=True)
    weights = AuxLossWeights(
        segmentation=0.0,
        paf=0.0,
        heatmap=1.0,
        heatmap_mse=0.3,
        heatmap_focal=0.4,
        heatmap_ridge=0.2,
    )

    losses = compute_aux_losses({"aux_maps": aux_maps}, targets, weights)

    assert losses["heatmap_coord"].item() == 0.0
    assert losses["heatmap_coord_var"].item() == 0.0
    assert losses["heatmap_peak"].item() == 0.0
    expected_heatmap_total = (
        weights.heatmap_mse * losses["heatmap_mse"]
        + weights.heatmap_focal * losses["heatmap_focal"]
        + weights.heatmap_ridge * losses["heatmap_ridge"]
    )
    assert torch.allclose(losses["heatmap_total"], expected_heatmap_total)
