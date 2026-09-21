import pytest
import torch

from kev.biencoder import CachedOptions, l2_normalize, masked_mean


def test_masked_mean_ignores_padding_tokens():
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    pooled = masked_mean(hidden, mask)

    assert pooled.tolist() == [[2.0, 4.0]]


def test_l2_normalize_returns_unit_vectors():
    vectors = torch.tensor([[3.0, 4.0], [5.0, 12.0]])

    normalized = l2_normalize(vectors)

    assert torch.linalg.vector_norm(normalized, dim=-1).tolist() == pytest.approx([1.0, 1.0])


def test_cached_options_reject_semantic_mismatch_even_when_length_matches():
    owner = object()
    cached = CachedOptions(owner, "pick a tool", ("browser", "shell"), torch.zeros(2, 4))

    cached.validate(owner, "pick a tool", ["browser", "shell"])
    with pytest.raises(ValueError, match="ordered option list"):
        cached.validate(owner, "pick a tool", ["shell", "browser"])
    with pytest.raises(ValueError, match="different question"):
        cached.validate(owner, "pick a route", ["browser", "shell"])
    with pytest.raises(ValueError, match="different model instance"):
        cached.validate(object(), "pick a tool", ["browser", "shell"])
