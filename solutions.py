"""Reference implementations: open AFTER attempting the exercise."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def feature_means(x):
    return x.mean(dim=0)


def split_heads(x, heads):
    if x.ndim != 3 or heads <= 0 or x.size(-1) % heads != 0:
        raise ValueError("Need [B,T,C], positive heads, and C divisible by heads")
    b, t, c = x.shape
    return x.reshape(b, t, heads, c // heads).transpose(1, 2)


def mse_weight_gradient(weight, x, y):
    w = weight.detach().clone().requires_grad_(True)
    loss = (w * x - y).square().mean()
    loss.backward()
    return w.grad.detach().clone()


def make_mlp(in_features, hidden, classes):
    return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, classes))


def standardize(train_x, other_x):
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, correction=0).clamp_min(1e-6)
    return (train_x - mean) / scale, (other_x - mean) / scale


def classification_loss(logits, targets):
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def evaluate_classifier(model, loader):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        count += len(y)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": total_loss / count, "accuracy": correct / count}


def masked_mean(embeddings, valid):
    weights = valid.unsqueeze(-1).to(embeddings.dtype)
    numerator = (embeddings * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1)
    return numerator / denominator


def language_model_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def causal_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    t = q.size(-2)
    allowed = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    return probabilities @ v
