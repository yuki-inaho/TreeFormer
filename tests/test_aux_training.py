import torch

from treeformer_train.aux_training import AuxLossWeights, build_aux_loss_weights, compute_aux_eval_metrics, compute_aux_losses


class Config:
    W_AUX_SEG = 2.0
    W_AUX_HEATMAP = 3.0
    W_AUX_PAF = 0.5


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

    assert weights == AuxLossWeights(segmentation=2.0, heatmap=3.0, paf=0.5)


def test_compute_aux_losses_backpropagates_with_resized_maps():
    output = {"aux_maps": torch.randn(2, 4, 4, 5, requires_grad=True)}
    losses = compute_aux_losses(output, _targets(), AuxLossWeights())

    assert set(losses) == {"total", "seg_bce", "heatmap_mse", "paf_l1"}
    assert losses["total"].ndim == 0

    losses["total"].backward()

    assert output["aux_maps"].grad is not None
    assert torch.isfinite(output["aux_maps"].grad).all()


def test_compute_aux_eval_metrics_reports_validation_signals():
    output = {"aux_maps": torch.zeros(2, 4, 8, 10)}
    metrics = compute_aux_eval_metrics(output, _targets(), AuxLossWeights())

    assert "total" in metrics
    assert "seg_iou" in metrics
    assert "heatmap_mae" in metrics
    assert "paf_masked_l1" in metrics
    assert 0.0 <= metrics["seg_iou"].item() <= 1.0
