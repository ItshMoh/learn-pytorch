"""Lesson 7: model mode and gradient recording are independent."""
import torch
from torch import nn


def main():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
    x = torch.ones(64, 4)
    for training in (True, False):
        model.train(training)
        for track_grad in (True, False):
            with torch.set_grad_enabled(track_grad):
                a, b = model(x), model(x)
            print("training:", training, "grad enabled:", track_grad,
                  "output requires_grad:", a.requires_grad,
                  "same outputs:", torch.equal(a, b))
    # Unequal batch sizes: average by examples, not by number of batches.
    batch_means = torch.tensor([1.0, 4.0])
    sizes = torch.tensor([4, 1])
    print("wrong epoch loss:", batch_means.mean().item())
    print("correct epoch loss:", ((batch_means * sizes).sum() / sizes.sum()).item())


if __name__ == "__main__":
    main()
