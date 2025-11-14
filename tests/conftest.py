"""Pytest fixtures for TreeFormer inference tests."""
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def dummy_model_output():
    """モデル出力のダミーデータを生成"""
    batch_size = 2
    num_tokens = 20
    return {
        'pred_logits': torch.randn(batch_size, num_tokens, 2),
        'pred_nodes': torch.rand(batch_size, num_tokens, 4)
    }


@pytest.fixture
def dummy_hidden_features():
    """隠れ層特徴のダミーデータを生成"""
    batch_size = 2
    seq_len = 21  # 20 obj + 1 rln
    hidden_dim = 256
    return torch.randn(batch_size, seq_len, hidden_dim)


@pytest.fixture
def mock_network():
    """モックニューラルネットワークを作成"""
    class MockModule:
        relation_embed = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    class MockNetwork:
        module = MockModule()

    return MockNetwork()
