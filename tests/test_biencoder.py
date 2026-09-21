import pytest
import torch

from kev.biencoder import l2_normalize, masked_mean


def test_masked_mean_ignores_padding_tokens():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = masked_mean(hidden, mask)

    assert pooled.tolist() == [[2.0, 4.0]]


def test_l2_normalize_returns_unit_vectors():
    vectors = torch.tensor([[3.0, 4.0], [5.0, 12.0]])

    normalized = l2_normalize(vectors)

    assert torch.linalg.vector_norm(normalized, dim=-1).tolist() == pytest.approx([1.0, 1.0])

