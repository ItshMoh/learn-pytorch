"""Lesson 4: fit y=3x-0.5 with two registered parameters."""
import torch
from torch import nn


class ScalarRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.weight + self.bias


def fit(steps=200):
    x = torch.linspace(-1, 1, 80)
    y = 3 * x - 0.5
    model = ScalarRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = (prediction - y).square().mean()
        loss.backward()
        optimizer.step()
    return model, loss.item()


def main():
    model, loss = fit()
    print("parameters:", list(model.named_parameters()))
    print("final loss:", loss)
    layer = nn.Linear(4, 6)
    print("linear weight:", layer.weight.shape, "bias:", layer.bias.shape)
    print("linear output for B,T,C:", layer(torch.zeros(2, 3, 4)).shape)


if __name__ == "__main__":
    main()
