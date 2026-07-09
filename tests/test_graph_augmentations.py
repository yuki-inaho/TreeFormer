import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from treeformer_train.augmentations import (
    ComposeGraphTransforms,
    ElasticGraphTransform,
    GraphSample,
    OpenCVPhotometricTransform,
    RandomAffineGraphTransform,
    build_graph_augmentation,
)


def _sample() -> GraphSample:
    x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    image = np.stack([grid_x, grid_y, np.full_like(grid_x, 0.5)], axis=2)
    nodes = np.array([[0.35, 0.35], [0.50, 0.52], [0.65, 0.64]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    return GraphSample(image=image, nodes=nodes, edges=edges)


def test_opencv_photometric_transform_preserves_graph_contract():
    transform = ComposeGraphTransforms(
        [
            OpenCVPhotometricTransform(
                p=1.0,
                brightness_contrast_p=1.0,
                hsv_p=1.0,
                gamma_p=1.0,
                noise_p=0.0,
                blur_p=0.0,
            )
        ],
        seed=7,
    )

    original = _sample()
    augmented = transform(original)

    assert augmented.image.shape == original.image.shape
    assert np.allclose(augmented.nodes, original.nodes)
    assert np.array_equal(augmented.edges, original.edges)
    assert not np.allclose(augmented.image, original.image)


def test_random_affine_transform_updates_nodes_with_topology_preserved():
    original = _sample()
    transform = RandomAffineGraphTransform(
        p=1.0,
        max_rotate_deg=0.0,
        max_translate_frac=0.05,
        scale_range=(1.0, 1.0),
        keep_all_nodes_inside=True,
    )

    augmented = transform(original, np.random.default_rng(3))

    assert augmented.image.shape == original.image.shape
    assert np.array_equal(augmented.edges, original.edges)
    assert np.all((0.0 <= augmented.nodes) & (augmented.nodes <= 1.0))
    assert not np.allclose(augmented.nodes, original.nodes)


def test_elastic_transform_updates_nodes_with_topology_preserved():
    original = _sample()
    transform = ElasticGraphTransform(
        p=1.0,
        alpha_frac=0.012,
        sigma_frac=0.04,
        grid_size=4,
        keep_all_nodes_inside=True,
    )

    augmented = transform(original, np.random.default_rng(11))

    assert augmented.image.shape == original.image.shape
    assert np.array_equal(augmented.edges, original.edges)
    assert np.all((0.0 <= augmented.nodes) & (augmented.nodes <= 1.0))
    assert not np.allclose(augmented.nodes, original.nodes)


def test_augmentation_factory_builds_composable_transform_from_config():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "seed": 3407,
            "photometric": {"enabled": True, "backend": "opencv", "p": 1.0},
            "affine": {
                "enabled": True,
                "p": 1.0,
                "max_rotate_deg": 0.0,
                "max_translate_frac": 0.02,
                "scale_range": [1.0, 1.0],
                "keep_all_nodes_inside": True,
            },
            "elastic": {"enabled": False},
        }
    )

    transform = build_graph_augmentation(cfg)
    assert transform is not None

    original = _sample()
    augmented = transform(original)

    assert augmented.image.shape == original.image.shape
    assert np.array_equal(augmented.edges, original.edges)
    assert np.all((0.0 <= augmented.nodes) & (augmented.nodes <= 1.0))


def test_legacy_loader_applies_graph_augmentation_without_changing_return_contract(tmp_path):
    from types import SimpleNamespace

    from train_mst import LoadCNNDataset

    split_root = tmp_path / "train"
    (split_root / "data").mkdir(parents=True)
    (split_root / "img").mkdir()
    Image.new("RGB", (64, 64), color=(80, 120, 160)).save(split_root / "img" / "sample.png")
    torch.save(
        SimpleNamespace(
            list_DETR_points_left_up=torch.tensor(
                [[0.35, 0.35], [0.50, 0.52], [0.65, 0.64]],
                dtype=torch.float32,
            ),
            DETR_node_collections=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ),
        split_root / "data" / "sample.pt",
    )
    sample_transform = ComposeGraphTransforms(
        [
            RandomAffineGraphTransform(
                p=1.0,
                max_rotate_deg=0.0,
                max_translate_frac=0.02,
                scale_range=(1.0, 1.0),
            )
        ],
        seed=17,
    )

    dataset = LoadCNNDataset(
        parent_path=split_root,
        max_size=64,
        is_train=False,
        is_rotate=False,
        sample_transform=sample_transform,
    )
    sample = dataset[0]

    assert isinstance(sample, tuple)
    assert len(sample) == 9
    image, name, nodes, edges, pafs, mask, unet, heatmap, sample_id = sample
    assert name == "sample"
    assert sample_id == "sample.pt"
    assert image.shape[0] == 3
    assert torch.all((nodes >= 0.0) & (nodes <= 1.0))
    assert not torch.allclose(
        nodes,
        torch.tensor([[0.35, 0.35], [0.50, 0.52], [0.65, 0.64]], dtype=torch.float32),
    )
    assert torch.equal(edges, torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    assert pafs.shape == (image.shape[1], image.shape[2], 2)
    assert mask.shape == image.shape[1:]
    assert unet.shape == image.shape[1:]
    assert heatmap.shape == image.shape[1:]
