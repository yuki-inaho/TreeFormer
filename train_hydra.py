from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from treeformer_train.checkpoint import CheckpointManager, load_pretrained_model_weights, load_training_checkpoint
from treeformer_train.config import as_plain_container, make_legacy_config
from treeformer_train.ema import ModelEma, unwrap_model
from treeformer_train.optimizers import build_optimizer_bundle, set_optimizer_eval_mode, set_optimizer_train_mode
from treeformer_train.runtime import barrier, setup_distributed, setup_reproducibility
from treeformer_train.tensorboard import TensorBoardLogger


@dataclass
class LegacyArgs:
    config: str | None = None
    resume: str | None = None
    device: str = "cuda"
    cuda_visible_device: tuple[int, ...] = (0,)
    use_gnn: bool = False
    use_mst_train: bool = True
    local_rank: int = 0


def _select_device(cfg: DictConfig, local_rank: int) -> torch.device:
    requested = str(cfg.runtime.device)
    if requested == "cuda":
        if not torch.cuda.is_available():
            if bool(cfg.runtime.fail_if_cuda_unavailable):
                raise RuntimeError("runtime.device=cuda was requested, but CUDA is unavailable. Set runtime.device=cpu explicitly for CPU-only checks.")
            raise RuntimeError("CUDA unavailable and implicit CPU fallback is disabled by project policy")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported runtime.device: {requested!r}")


def _build_dataloaders(legacy_config: Any, distributed_context: Any) -> tuple[DataLoader, DataLoader]:
    from train_mst import build_train_val_datasets, custom_collate_fn

    dataset_train, dataset_val = build_train_val_datasets(legacy_config.DATA)
    num_workers = int(getattr(legacy_config.DATA, "NUM_WORKERS", 0))
    if distributed_context.is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        valid_sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset_train,
        batch_size=int(legacy_config.DATA.BATCH_SIZE),
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=int(legacy_config.DATA.BATCH_SIZE),
        shuffle=False,
        collate_fn=custom_collate_fn,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        sampler=valid_sampler,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "training dataloader has zero batches; increase DATA.TRAIN_LIMIT, use a larger training split, "
            "or reduce DATA.BATCH_SIZE"
        )
    return train_loader, val_loader


def _load_resume_if_requested(
    checkpoint_path: str | None,
    *,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    ema: ModelEma | None,
    map_location: torch.device,
) -> int:
    if not checkpoint_path:
        return 1
    checkpoint = load_training_checkpoint(checkpoint_path, map_location=map_location)
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint.get("scheduler", {}))
    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    return int(checkpoint["epoch"]) + 1


def _metrics_dict(
    *,
    train_total: float,
    train_class: float,
    train_nodes: float,
    train_edges: float,
    train_boxes: float,
    train_cards: float,
    val_smd: float,
    lr: float,
    epoch_seconds: float,
    best_metric: float | None,
) -> dict[str, float]:
    metrics = {
        "train/total_loss": train_total,
        "train/class_loss": train_class,
        "train/nodes_loss": train_nodes,
        "train/edges_loss": train_edges,
        "train/boxes_loss": train_boxes,
        "train/cards_loss": train_cards,
        "val/smd": val_smd,
        "optim/lr": lr,
        "time/epoch_seconds": epoch_seconds,
    }
    if best_metric is not None:
        metrics["checkpoint/best_metric"] = best_metric
    return metrics


def _is_aux_supervised_training(cfg: DictConfig) -> bool:
    mode = str(OmegaConf.select(cfg, "TRAIN.MODE", default="graph"))
    return mode == "aux_supervised" or bool(OmegaConf.select(cfg, "TRAIN.SKIP_GRAPH_OUTPUT", default=False))


def _aux_metrics_dict(
    *,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    epoch_seconds: float,
    best_metric: float | None,
) -> dict[str, float]:
    metrics = {
        "train/aux_total_loss": train_metrics["total"],
        "train/seg_total_loss": train_metrics["seg_total"],
        "train/aux_seg_bce": train_metrics["seg_bce"],
        "train/aux_seg_dice_loss": train_metrics["seg_dice"],
        "train/aux_seg_focal_loss": train_metrics["seg_focal"],
        "train/aux_heatmap_mse": train_metrics["heatmap_mse"],
        "train/aux_paf_l1": train_metrics["paf_l1"],
        "val/aux_total_loss": val_metrics["total"],
        "val/seg_total_loss": val_metrics["seg_total"],
        "val/aux_seg_bce": val_metrics["seg_bce"],
        "val/aux_seg_dice_loss": val_metrics["seg_dice"],
        "val/aux_seg_focal_loss": val_metrics["seg_focal"],
        "val/aux_heatmap_mse": val_metrics["heatmap_mse"],
        "val/aux_paf_l1": val_metrics["paf_l1"],
        "val/seg_iou": val_metrics["seg_iou"],
        "val/seg_dice_score": val_metrics["seg_dice_score"],
        "val/seg_soft_dice_score": val_metrics["seg_soft_dice_score"],
        "val/seg_precision": val_metrics["seg_precision"],
        "val/seg_recall": val_metrics["seg_recall"],
        "val/pred_positive_rate": val_metrics["pred_positive_rate"],
        "val/target_positive_rate": val_metrics["target_positive_rate"],
        "val/heatmap_mae": val_metrics["heatmap_mae"],
        "val/paf_masked_l1": val_metrics["paf_masked_l1"],
        "optim/lr": lr,
        "time/epoch_seconds": epoch_seconds,
    }
    if best_metric is not None:
        metrics["checkpoint/best_metric"] = best_metric
    return metrics


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    aux_supervised_training = _is_aux_supervised_training(cfg)
    if aux_supervised_training:
        cfg.MODEL.GRAPH_OUTPUT_ENABLED = False

    legacy_config = make_legacy_config(cfg)
    distributed_context = setup_distributed(cfg.distributed)
    device = _select_device(cfg, distributed_context.local_rank)
    setup_reproducibility(int(legacy_config.DATA.SEED) + distributed_context.rank)

    if distributed_context.is_rank_zero:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(legacy_config.log.message)

    from models import build_model

    train_loader, val_loader = _build_dataloaders(legacy_config, distributed_context)
    if distributed_context.is_rank_zero:
        print(f"Dataset splits -> Train: {len(train_loader.dataset)} | Valid: {len(val_loader.dataset)}")

    if aux_supervised_training and not bool(OmegaConf.select(cfg, "MODEL.AUX_HEAD.ENABLED", default=False)):
        raise ValueError("aux supervised training requires MODEL.AUX_HEAD.ENABLED=true")

    args = LegacyArgs(device=str(cfg.runtime.device), use_mst_train=True, local_rank=distributed_context.local_rank)
    model = build_model(legacy_config, args).to(device)
    load_pretrained_model_weights(
        model,
        cfg.checkpoint.pretrained,
        key=str(cfg.checkpoint.pretrained_key),
        strict=bool(cfg.checkpoint.pretrained_strict),
        map_location=device,
    )

    optimizer_bundle = build_optimizer_bundle(model, legacy_config.TRAIN, cfg.optimizer)
    output_dir = Path(str(OmegaConf.select(cfg, "runtime.output_dir")))
    if distributed_context.is_rank_zero:
        optimizer_bundle.write_parameter_report(output_dir / str(cfg.runtime.parameter_report_name))

    if bool(cfg.ema.enabled):
        ema_device = cfg.ema.device if cfg.ema.device is not None else None
        ema = ModelEma(model, decay=float(cfg.ema.decay), device=ema_device)
    else:
        ema = None

    if distributed_context.is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[distributed_context.local_rank] if device.type == "cuda" else None,
            output_device=distributed_context.local_rank if device.type == "cuda" else None,
            find_unused_parameters=aux_supervised_training,
        )

    if aux_supervised_training:
        from treeformer_train.aux_training import build_aux_loss_weights, epoch_train_aux, epoch_val_aux

        aux_loss_weights = build_aux_loss_weights(legacy_config.TRAIN)
        loss = None
        smd = None
    else:
        from epoch import epoch_train, epoch_val
        from losses_only import SetCriterion
        from metric_smd import StreetMoverDistance
        from models.matcher import build_matcher
        from monai.utils import MetricReduction

        matcher = build_matcher(legacy_config)
        loss = SetCriterion(config=legacy_config, matcher=matcher, net=model, args=args)
        smd = StreetMoverDistance(eps=1e-7, max_iter=100, reduction=MetricReduction.MEAN)
    checkpoint_manager = CheckpointManager(
        cfg.checkpoint.dir,
        metric_name=str(cfg.checkpoint.metric_name),
        mode=str(cfg.checkpoint.mode),
        save_last=bool(cfg.checkpoint.save_last),
        save_best=bool(cfg.checkpoint.save_best),
        save_every=int(cfg.checkpoint.save_every),
    )
    writer = TensorBoardLogger(
        cfg.tensorboard.log_dir,
        enabled=bool(cfg.tensorboard.enabled),
        rank=distributed_context.rank,
        flush_secs=int(cfg.tensorboard.flush_secs),
    )

    start_epoch = _load_resume_if_requested(
        cfg.checkpoint.resume,
        model=model,
        optimizer=optimizer_bundle.optimizer,
        scheduler=optimizer_bundle.scheduler,
        ema=ema,
        map_location=device,
    )
    max_epochs = int(legacy_config.TRAIN.EPOCHS)
    if max_epochs <= 0:
        if distributed_context.is_rank_zero:
            print("Hydra dry-run completed: dataset, dataloader, model, optimizer, scheduler, EMA and checkpoint manager initialized.")
        writer.close()
        barrier(distributed_context)
        return

    for epoch in range(start_epoch, max_epochs + 1):
        if distributed_context.is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        set_optimizer_train_mode(optimizer_bundle.optimizer, required=optimizer_bundle.requires_train_eval)
        if aux_supervised_training:
            train_metrics = epoch_train_aux(
                train_loader=train_loader,
                net=model,
                optimizer=optimizer_bundle.optimizer,
                device=device,
                epoch_now=epoch,
                max_epoch=max_epochs,
                loss_weights=aux_loss_weights,
                clip_max_norm=float(legacy_config.TRAIN.CLIP_MAX_NORM),
                after_optimizer_step=(lambda **_: ema.update(model)) if ema is not None else None,
            )
        else:
            train_total, train_class, train_nodes, train_edges, train_boxes, train_cards = epoch_train(
                train_loader=train_loader,
                net=model,
                loss_function=loss,
                optimizer=optimizer_bundle.optimizer,
                device=device,
                last_epoch=start_epoch,
                epoch_now=epoch,
                max_epoch=max_epochs,
                clip_max_norm=float(legacy_config.TRAIN.CLIP_MAX_NORM),
                after_optimizer_step=(lambda **_: ema.update(model)) if ema is not None else None,
            )

        set_optimizer_eval_mode(optimizer_bundle.optimizer, required=optimizer_bundle.requires_train_eval)
        if aux_supervised_training:
            if ema is not None and bool(cfg.ema.evaluate):
                with ema.average_parameters(model):
                    val_metrics = epoch_val_aux(val_loader=val_loader, net=model, device=device, loss_weights=aux_loss_weights)
            else:
                val_metrics = epoch_val_aux(val_loader=val_loader, net=model, device=device, loss_weights=aux_loss_weights)
        else:
            if ema is not None and bool(cfg.ema.evaluate):
                with ema.average_parameters(model):
                    val_smd = epoch_val(val_loader=val_loader, net=model, config=legacy_config, device=device, SMD=smd, args=args)
            else:
                val_smd = epoch_val(val_loader=val_loader, net=model, config=legacy_config, device=device, SMD=smd, args=args)

        epoch_seconds = time.time() - epoch_start
        lr = float(optimizer_bundle.optimizer.param_groups[0]["lr"])
        if aux_supervised_training:
            metrics = _aux_metrics_dict(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=lr,
                epoch_seconds=epoch_seconds,
                best_metric=checkpoint_manager.best_metric,
            )
        else:
            metrics = _metrics_dict(
                train_total=train_total,
                train_class=train_class,
                train_nodes=train_nodes,
                train_edges=train_edges,
                train_boxes=train_boxes,
                train_cards=train_cards,
                val_smd=val_smd,
                lr=lr,
                epoch_seconds=epoch_seconds,
                best_metric=checkpoint_manager.best_metric,
            )

        if distributed_context.is_rank_zero:
            writer.add_scalars(epoch, metrics)
            checkpoint_result = None
            if bool(cfg.checkpoint.enabled):
                checkpoint_result = checkpoint_manager.save(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer_bundle.optimizer,
                    scheduler=optimizer_bundle.scheduler,
                    metrics=metrics,
                    ema=ema,
                    config=as_plain_container(cfg),
                    extra={"parameter_report": str(output_dir / str(cfg.runtime.parameter_report_name))},
                )
            best_summary = (
                f" | best={checkpoint_result.best_metric} at epoch {checkpoint_result.best_epoch}"
                if checkpoint_result is not None
                else ""
            )
            if aux_supervised_training:
                print(
                    f"Epoch {epoch}/{max_epochs} | train_aux_total={train_metrics['total']:.6f} "
                    f"| val_aux_total={val_metrics['total']:.8f} | val_seg_iou={val_metrics['seg_iou']:.6f} "
                    f"| val_seg_dice={val_metrics['seg_dice_score']:.6f}{best_summary}"
                )
            else:
                print(f"Epoch {epoch}/{max_epochs} | train_total={train_total:.6f} | val_smd={val_smd:.8f}{best_summary}")
        optimizer_bundle.scheduler.step()
        barrier(distributed_context)

    writer.close()
    if distributed_context.is_rank_zero:
        print("Hydra training completed.")


if __name__ == "__main__":
    main()
