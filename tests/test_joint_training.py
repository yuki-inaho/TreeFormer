import torch

from treeformer_train.aux_training import AuxLossComputer, AuxLossWeights
from treeformer_train.joint_training import epoch_train_joint_graph_aux


class TinyJointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aux = torch.nn.Parameter(torch.zeros(1, 5, 4, 4))

    def forward(self, images):
        batch_size = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        return None, {"aux_maps": self.aux.expand(batch_size, -1, -1, -1)}


class TinyGraphLoss:
    def __call__(self, _h, output, target, _epoch_now, _max_epoch, _last_epoch):
        total = output["aux_maps"].mean().pow(2) + 0.1 * len(target["nodes"])
        zero = total.new_zeros(())
        return {
            "total": total,
            "class": zero,
            "nodes": zero,
            "edges": zero,
            "boxes": zero,
            "cards": zero,
        }


def _batch():
    image = torch.zeros(3, 4, 4)
    node = torch.zeros(1, 2)
    edge = torch.zeros(0, 2)
    paf = torch.zeros(1, 2, 4, 4)
    paf_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    segmentation = torch.zeros(1, 1, 4, 4)
    segmentation[:, :, 1:3, 1:3] = 1.0
    heatmap = torch.zeros(1, 1, 4, 4)
    heatmap[:, :, 2, 2] = 1.0
    return ([ [image], [node], [edge], paf, paf_mask, segmentation, heatmap, ["sample"] ],)


def test_epoch_train_joint_graph_aux_backpropagates_graph_and_aux_losses():
    model = TinyJointModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = epoch_train_joint_graph_aux(
        train_loader=[_batch()],
        net=model,
        graph_loss_function=TinyGraphLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        last_epoch=1,
        epoch_now=1,
        max_epoch=1,
        aux_loss_weights=AuxLossWeights(detail=0.1, heatmap=1.0, paf=0.25),
        aux_loss_computer=AuxLossComputer(AuxLossWeights(detail=0.1, heatmap=1.0, paf=0.25)),
        joint_aux_weight=0.5,
    )

    assert metrics["joint_total"] > metrics["graph_total"]
    assert metrics["aux_total"] > 0.0
    assert torch.isfinite(model.aux).all()
