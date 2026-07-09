import torch

from models.utils import nested_tensor_from_tensor_list


def test_nested_tensor_accepts_batched_tensor_without_padding():
    images = torch.randn(3, 3, 16, 24)

    nested = nested_tensor_from_tensor_list(images)
    tensors, mask = nested.decompose()

    assert tensors is images
    assert mask.shape == (3, 16, 24)
    assert mask.dtype == torch.bool
    assert not mask.any()
