import torch
import pytest
from PIL import Image


def _minimal_annotation():
    return {
        "VineImage": [
            {
                "VineFeature": [
                    [
                        {
                            "FeatureID": 10,
                            "ParentID": None,
                            "FeatureCoordinates": [10, 5],
                        },
                        {
                            "FeatureID": 11,
                            "ParentID": 10,
                            "FeatureCoordinates": [20, 15],
                        },
                    ]
                ]
            }
        ]
    }


def _write_sample(root, *, with_annotation=True):
    split_dir = root / "01-TrainAndValidationSet"
    split_dir.mkdir(parents=True)
    image_path = split_dir / "sample.jpeg"
    Image.new("RGB", (100, 50), color=(64, 128, 255)).save(image_path)

    if with_annotation:
        annotation_path = split_dir / "sample_annotation.json"
        annotation_path.write_text(
            __import__("json").dumps(_minimal_annotation()),
            encoding="utf-8",
        )

    return image_path


def test_parse_single_annotation():
    from guyot_dataset import parse_guyot_annotation

    annotation = {
        "VineImage": [
            {
                "VineFeature": [
                    [
                        {
                            "FeatureID": 10,
                            "ParentID": None,
                            "FeatureCoordinates": [100, 50],
                        },
                        {
                            "FeatureID": 11,
                            "ParentID": 10,
                            "FeatureCoordinates": [200, 150],
                        },
                        {
                            "FeatureID": 12,
                            "ParentID": 11,
                            "FeatureCoordinates": [300, 250],
                        },
                        {
                            "FeatureID": 13,
                            "ParentID": 13,
                            "FeatureCoordinates": [400, 350],
                        },
                    ]
                ]
            }
        ]
    }

    nodes, edges = parse_guyot_annotation(annotation, image_size=(1000, 500))

    assert isinstance(nodes, torch.Tensor)
    assert isinstance(edges, torch.Tensor)
    assert nodes.dtype == torch.float32
    assert edges.dtype == torch.long
    assert nodes.shape == (4, 2)
    assert edges.shape == (2, 2)

    expected_nodes = torch.tensor(
        [
            [0.1, 0.1],
            [0.2, 0.3],
            [0.3, 0.5],
            [0.4, 0.7],
        ],
        dtype=torch.float32,
    )
    expected_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    assert torch.allclose(nodes, expected_nodes)
    assert torch.equal(edges, expected_edges)
    assert torch.all((nodes >= 0.0) & (nodes <= 1.0))


def test_dataset_returns_training_sample(tmp_path):
    from guyot_dataset import GuyotDataset

    _write_sample(tmp_path)

    dataset = GuyotDataset(root=tmp_path, split="train")
    sample = dataset[0]

    assert set(sample) == {"image", "nodes", "edges", "filename"}
    assert sample["filename"] == "sample.jpeg"
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (3, 50, 100)
    assert torch.all((sample["image"] >= 0.0) & (sample["image"] <= 1.0))

    assert torch.allclose(
        sample["nodes"],
        torch.tensor([[0.1, 0.1], [0.2, 0.3]], dtype=torch.float32),
    )
    assert torch.equal(sample["edges"], torch.tensor([[0, 1]], dtype=torch.long))


def test_missing_annotation_raises(tmp_path):
    from guyot_dataset import GuyotDataset

    _write_sample(tmp_path, with_annotation=False)

    dataset = GuyotDataset(root=tmp_path, split="train")
    with pytest.raises(FileNotFoundError, match="annotation"):
        dataset[0]


def test_training_adapter_returns_legacy_sample_contract(tmp_path):
    from guyot_dataset import GuyotDataset, GuyotTrainingAdapter

    _write_sample(tmp_path)

    dataset = GuyotTrainingAdapter(GuyotDataset(root=tmp_path, split="train"), max_size=64)
    sample = dataset[0]

    assert isinstance(sample, tuple)
    assert len(sample) == 9

    image, name, nodes, edges, pafs, mask, unet, heatmap, sample_id = sample
    assert name == "sample"
    assert sample_id == "sample.jpeg"

    assert image.dtype == torch.float32
    assert image.shape[0] == 3
    assert max(image.shape[1:]) <= 64
    assert torch.all((image >= -1.0) & (image <= 1.0))

    assert nodes.dtype == torch.float32
    assert torch.allclose(nodes, torch.tensor([[0.1, 0.1], [0.2, 0.3]], dtype=torch.float32))
    assert edges.dtype == torch.long
    assert torch.equal(edges, torch.tensor([[0, 1]], dtype=torch.long))

    assert pafs.dtype == torch.float32
    assert pafs.shape == (image.shape[1], image.shape[2], 2)
    assert torch.all((pafs >= -1.0) & (pafs <= 1.0))

    assert mask.dtype == torch.bool
    assert mask.shape == image.shape[1:]

    assert unet.dtype == torch.float32
    assert unet.shape == image.shape[1:]
    assert torch.all((unet >= 0.0) & (unet <= 1.0))

    assert heatmap.dtype == torch.float32
    assert heatmap.shape == image.shape[1:]
    assert torch.all((heatmap >= 0.0) & (heatmap <= 1.0))
