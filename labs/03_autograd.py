"""Lesson 3: backward computes gradients; it does not update weights."""
import torch


def main():
    w = torch.tensor(1.0, requires_grad=True)
    loss = (2 * w - 5).square()
    loss.backward()
    print("loss, gradient:", loss.item(), w.grad.item())  # 9, -12
    with torch.no_grad():
        w -= 0.1 * w.grad
    print("updated weight:", w.item())  # 2.2
    w.grad = None
    # A fresh forward graph each time; .grad still accumulates across backwards.
    (w.square()).backward()
    first = w.grad.clone()
    (w.square()).backward()
    torch.testing.assert_close(w.grad, 2 * first)
    print("accumulated gradient:", w.grad.item())
    independent = w.detach().clone()
    print("detached copy requires grad:", independent.requires_grad)


if __name__ == "__main__":
    main()
