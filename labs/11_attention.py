"""Lesson 11: verify mathematics, causality, and gradients separately."""
import torch
from torch.nn import functional as F
from attention import causal_attention, CausalSelfAttention


def verify():
    torch.manual_seed(19)
    q, k, v = (torch.randn(2, 3, 5, 4) for _ in range(3))
    context, probabilities = causal_attention(q, k, v)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    torch.testing.assert_close(context, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, 3, 5))
    assert torch.count_nonzero(probabilities.triu(diagonal=1)) == 0
    model = CausalSelfAttention(channels=12, heads=3)
    x = torch.randn(2, 5, 12)
    modified = x.clone()
    modified[:, 3:] = torch.randn_like(modified[:, 3:]) * 10
    original = model(x)
    torch.testing.assert_close(original[:, :3], model(modified)[:, :3])
    original.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return context.shape


if __name__ == "__main__":
    print("Verified manual attention:", verify())
