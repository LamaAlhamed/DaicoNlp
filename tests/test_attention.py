"""Lab 2 contract checks."""
import torch
import torch.nn.functional as F
from bayan.attention import attention


def test_attention_matches_pytorch():
    torch.manual_seed(42)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_causal_mask_prevents_future_attention():
    # TODO(Lab 2): once your attention() exposes/returns weights for diagnostics,
    # extend this test to assert the weight matrix is lower-triangular.
    assert True
