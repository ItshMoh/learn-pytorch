"""Manual causal multi-head attention. Lesson 11 explains every operation."""
import math
import torch
from torch import nn


def causal_attention(q, k, v):
    """q,k,v: [B,H,T,D]. Return context and attention probabilities."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("This teaching version requires equal Q/K/V shapes")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    t = q.size(-2)
    allowed = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = scores.softmax(dim=-1)
    return probabilities @ v, probabilities


class CausalSelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        b, t, c = x.shape
        d = c // self.heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, t, self.heads, d).transpose(1, 2)
        k = k.reshape(b, t, self.heads, d).transpose(1, 2)
        v = v.reshape(b, t, self.heads, d).transpose(1, 2)
        context, _ = causal_attention(q, k, v)
        merged = context.transpose(1, 2).reshape(b, t, c)
        return self.out(merged)
