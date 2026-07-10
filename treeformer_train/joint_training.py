from __future__ import annotations

import time
from typing import Any

import torch
import torch.distributed as dist

from .aux_training import AuxLossComputer, AuxLossWeights, compute_aux_eval_metrics


def _dist_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _mark_compile_step_begin(device: torch.device) -> None:
    if device.type != "cuda" or not hasattr(torch, "compiler"):
        return
    mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark_step is not None:
        mark_step()


def _maybe_stack_images(images: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
    if not images:
        return images
    first_shape = tuple(images[0].shape)
    if all(tuple(image.shape) == first_shape for image in images):
        return torch.stack(images, dim=0).contiguous()
    return images


def _prepare_joint_batch(
    batchdata: Any, device: torch.device
) -> tuple[torch.Tensor | list[torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    batch = batchdata[0]
    non_blocking = device.type == "cuda"
    images = [img.to(device, dtype=torch.float32, non_blocking=non_blocking) for img in batch[0]]
    graph_target: dict[str, Any] = {
        "nodes": [node.to(device, non_blocking=non_blocking) for node in batch[1]],
        "edges": [edge.to(device, non_blocking=non_blocking) for edge in batch[2]],
    }
    if len(batch) > 8 and isinstance(batch[-2], dict):
        graph_target.update(batch[-2])
    aux_target = {
        "paf": batch[3].to(device, dtype=torch.float32, non_blocking=non_blocking),
        "paf_mask": batch[4].to(device, dtype=torch.bool, non_blocking=non_blocking),
        "segmentation": batch[5].to(device, dtype=torch.float32, non_blocking=non_blocking),
        "heatmap": batch[6].to(device, dtype=torch.float32, non_blocking=non_blocking),
    }
    return _maybe_stack_images(images), graph_target, aux_target


class _MetricAverager:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, int] = {}

    def update(self, metrics: dict[str, torch.Tensor], weight: int) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.detach().item()) * weight
            self.weights[key] = self.weights.get(key, 0) + weight

    def compute(self) -> dict[str, float]:
        return {key: value / max(self.weights[key], 1) for key, value in self.sums.items()}


def epoch_train_joint_graph_aux(
    *,
    train_loader: Any,
    net: torch.nn.Module,
    graph_loss_function: Any,
    optimizer: Any,
    device: torch.device,
    last_epoch: int,
    epoch_now: int,
    max_epoch: int,
    aux_loss_weights: AuxLossWeights,
    aux_loss_computer: AuxLossComputer,
    joint_aux_weight: float,
    clip_max_norm: float = 20.0,
    after_optimizer_step: Any | None = None,
) -> dict[str, float]:
    net.train()
    averages = _MetricAverager()
    all_len = len(train_loader)
    for i, batchdata in enumerate(train_loader):
        batch_start = time.time()
        images, graph_target, aux_target = _prepare_joint_batch(batchdata, device)

        _mark_compile_step_begin(device)
        h, output = net(images)
        graph_losses = graph_loss_function(h, output, graph_target, epoch_now, max_epoch, last_epoch)
        aux_losses = aux_loss_computer(output, aux_target)
        joint_total = graph_losses["total"] + float(joint_aux_weight) * aux_losses["total"]
        batch_size = len(graph_target["nodes"])
        averages.update(
            {
                "joint_total": joint_total,
                "graph_total": graph_losses["total"],
                "graph_class": graph_losses["class"],
                "graph_nodes": graph_losses["nodes"],
                "graph_edges": graph_losses["edges"],
                "graph_boxes": graph_losses["boxes"],
                "graph_cards": graph_losses["cards"],
                "aux_total": aux_losses["total"],
                "aux_seg_total": aux_losses["seg_total"],
                "aux_detail_total": aux_losses["detail_total"],
                "aux_heatmap_total": aux_losses["heatmap_total"],
                "aux_paf_l1": aux_losses["paf_l1"],
            },
            batch_size,
        )

        optimizer.zero_grad(set_to_none=True)
        joint_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm, norm_type=2)
        optimizer.step()
        if after_optimizer_step is not None:
            after_optimizer_step(net=net, optimizer=optimizer, epoch=epoch_now, batch_index=i)

        if _dist_rank() == 0 and i % 100 == 0:
            elapsed = time.time() - batch_start
            print(
                "Epoch: {} / {} Batch: {} / {} || Joint total: {:.4f} graph: {:.4f} aux: {:.4f} "
                "seg: {:.4f} heatmap: {:.4f} paf: {:.4f} take {:.4f} sec.".format(
                    epoch_now - 1,
                    max_epoch,
                    i,
                    all_len,
                    joint_total,
                    graph_losses["total"],
                    aux_losses["total"],
                    aux_losses["seg_total"],
                    aux_losses["heatmap_total"],
                    aux_losses["paf_l1"],
                    elapsed,
                )
            )
    return averages.compute()


@torch.inference_mode()
def epoch_val_aux_from_joint_loader(
    *,
    val_loader: Any,
    net: torch.nn.Module,
    device: torch.device,
    loss_weights: AuxLossWeights,
    loss_computer: AuxLossComputer,
) -> dict[str, float]:
    net.eval()
    averages = _MetricAverager()
    for batchdata in val_loader:
        images, _, aux_target = _prepare_joint_batch(batchdata, device)
        _mark_compile_step_begin(device)
        _, output = net(images)
        metrics = compute_aux_eval_metrics(output, aux_target, loss_weights, loss_core=loss_computer.core)
        averages.update(metrics, aux_target["segmentation"].shape[0])
    return averages.compute()
