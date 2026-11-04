"""Lesson 9: unequal microbatches and gradient diagnostics."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def accumulation_demo():
    torch.manual_seed(2)
    x, y = torch.randn(7, 3), torch.randn(7, 1)
    full = nn.Linear(3, 1)
    micro = copy.deepcopy(full)
    F.mse_loss(full(x), y).backward()
    # 7 examples split as 3,3,1; weighting means equally would be wrong.
    for xb, yb in zip(x.split(3), y.split(3)):
        loss = F.mse_loss(micro(xb), yb)
        (loss * len(xb) / len(x)).backward()
    for a, b in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    return full


def main():
    model = accumulation_demo()
    for name, parameter in model.named_parameters():
        print(name, "gradient norm:", parameter.grad.norm().item())
        assert torch.isfinite(parameter.grad).all()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    print("norm BEFORE clipping:", norm.item())
    bad_prediction, target = torch.zeros(4, 1), torch.zeros(4)
    if bad_prediction.shape != target.shape:
        print("Caught a silent-broadcast bug before computing MSE")


if __name__ == "__main__":
    main()
