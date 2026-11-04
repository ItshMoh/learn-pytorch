"""Your implementations. Use check_exercises.py for feedback.

Work on one function after its lesson. Do not import solutions here.
"""
import torch
from torch import nn
from torch.nn import functional as F


def feature_means(x):
    return x.mean(dim=0)


def split_heads(x, heads):
    """2 / Lesson 2: [B,T,C] -> [B,H,T,C/H].

    Raise ValueError for non-3D input, nonpositive heads, or C not divisible by H.
    Preserve which channels belong to each token. No copying to Python lists.
    """
    raise NotImplementedError("Exercise 2: split channels, then exchange axes")


def mse_weight_gradient(weight, x, y):
    """3 / Lesson 3: use autograd for d mean((w*x-y)^2) / dw.

    weight is a scalar float tensor; x and y have the same shape. Return a scalar
    tensor. Use a fresh trainable copy; do not change weight or its .grad.
    """
    raise NotImplementedError("Exercise 3: create a leaf, compute loss, backward")


def make_mlp(in_features, hidden, classes):
    """4 / Lesson 4: return Linear -> ReLU -> Linear, with default biases."""
    raise NotImplementedError("Exercise 4: return an nn.Sequential")


def standardize(train_x, other_x):
    """5 / Lesson 5: return (normalized_train, normalized_other).

    Use ONLY train_x for per-feature mean and population std (correction=0).
    Clamp std to at least 1e-6, including for a constant feature.
    """
    raise NotImplementedError("Exercise 5: fit on train, apply to both")


def classification_loss(logits, targets):
    """6 / Lesson 6: mean multiclass CE for [B,K] logits and [B] long targets."""
    raise NotImplementedError("Exercise 6: logits are not probabilities")


def evaluate_classifier(model, loader):
    """7 / Lesson 7: CPU evaluation; return {'loss': float, 'accuracy': float}.

    Set eval mode, disable grad, weight loss by examples, and leave model in eval.
    Raise ValueError for an empty loader. Ordinary unweighted CE, no ignored labels.
    """
    raise NotImplementedError("Exercise 7: count examples, not batches")


def masked_mean(embeddings, valid):
    """8 / Lesson 10: [B,T,C] float + [B,T] bool -> [B,C].

    Average only valid positions. An all-padding example must return zeros.
    Remain differentiable with respect to embeddings.
    """
    raise NotImplementedError("Exercise 8: sum valid vectors and divide by valid count")


def language_model_loss(logits, targets):
    """9 / Lesson 10: [B,T,V] float + [B,T] long -> scalar mean token CE.

    These examples have no padding. Preserve vocabulary as the class dimension.
    """
    raise NotImplementedError("Exercise 9: flatten B and T, retain V")


def causal_attention(q, k, v):
    """10 / Lesson 11: [B,H,T,D] tensors -> context [B,H,T,D].

    Equal Q/K/V shapes, T>=1. Use matmul, scaling, an additive/masked score
    matrix, softmax, matmul. No SDPA or MultiheadAttention. No dropout needed.
    """
    raise NotImplementedError("Exercise 10: block future scores BEFORE softmax")
