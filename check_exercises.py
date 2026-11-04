"""Behavioral checks, not a substitute for explaining your implementation."""
import argparse
import importlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def check(number, implementation):
    torch.manual_seed(91)
    m = implementation
    if number == 1:
        x = torch.tensor([[1.0, 2.0, 9.0], [3.0, 4.0, 1.0]])
        torch.testing.assert_close(m.feature_means(x), torch.tensor([2.0, 3.0, 5.0]))
    elif number == 2:
        x = torch.arange(2 * 5 * 12.0).reshape(2, 5, 12)
        result = m.split_heads(x, 3)
        assert result.shape == (2, 3, 5, 4), "Expected [B,H,T,D]"
        for b in range(2):
            for t in range(5):
                for h in range(3):
                    torch.testing.assert_close(result[b, h, t], x[b, t, h*4:(h+1)*4])
        for invalid, heads in ((x, 5), (x, 0), (x[0], 3)):
            try:
                m.split_heads(invalid, heads)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid head configuration should raise ValueError")
    elif number == 3:
        w = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        x = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64)
        y = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
        actual = m.mse_weight_gradient(w, x, y)
        analytic = (2 * (w.detach() * x - y) * x).mean()
        eps = 1e-5
        loss = lambda a: ((a * x - y)**2).mean()
        numerical = (loss(w.detach()+eps) - loss(w.detach()-eps)) / (2*eps)
        torch.testing.assert_close(actual, analytic)
        torch.testing.assert_close(actual, numerical)
        assert w.grad is None and w.item() == 0.7, "Do not mutate the caller's tensor"
    elif number == 4:
        model = m.make_mlp(4, 5, 3)
        assert model(torch.randn(7, 4)).shape == (7, 3)
        assert sum(p.numel() for p in model.parameters()) == 43
        assert any(isinstance(layer, nn.ReLU) for layer in model.modules())
        model(torch.randn(2, 4)).square().sum().backward()
        assert all(p.grad is not None for p in model.parameters())
    elif number == 5:
        train = torch.tensor([[1.0, 7.0], [3.0, 7.0]])
        other = torch.tensor([[5.0, 7.0]])
        a, b = m.standardize(train, other)
        torch.testing.assert_close(a, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        torch.testing.assert_close(b, torch.tensor([[3.0, 0.0]]))
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
    elif number == 6:
        logits = torch.tensor([[1000.0, 999.0, 998.0], [-2.0, 1.0, 3.0]], requires_grad=True)
        y = torch.tensor([0, 2])
        actual = m.classification_loss(logits, y)
        expected = (-logits.log_softmax(dim=-1)[torch.arange(2), y]).mean()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    elif number == 7:
        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 3)
                self.observed = []
            def forward(self, x):
                self.observed.append((self.training, torch.is_grad_enabled()))
                return self.layer(x)
        model = Recorder()
        x, y = torch.randn(5, 2), torch.tensor([0, 2, 1, 0, 2])
        with torch.no_grad():
            all_logits = model.layer(x)
            expected_loss = F.cross_entropy(all_logits, y).item()
            expected_acc = (all_logits.argmax(-1) == y).float().mean().item()
        result = m.evaluate_classifier(model, DataLoader(TensorDataset(x, y), batch_size=4))
        assert math.isclose(result["loss"], expected_loss, rel_tol=1e-6)
        assert math.isclose(result["accuracy"], expected_acc, abs_tol=1e-7)
        assert all(not training and not grad for training, grad in model.observed)
        assert all(p.grad is None for p in model.parameters())
        try:
            m.evaluate_classifier(model, DataLoader(TensorDataset(x[:0], y[:0]), batch_size=4))
        except ValueError:
            pass
        else:
            raise AssertionError("Empty evaluation should raise ValueError")
    elif number == 8:
        embeddings = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [99.0, 99.0]],
                                   [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]], requires_grad=True)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        output = m.masked_mean(embeddings, valid)
        torch.testing.assert_close(output, torch.tensor([[3.0, 6.0], [0.0, 0.0]]))
        output.sum().backward()
        assert embeddings.grad is not None
        assert torch.count_nonzero(embeddings.grad[~valid]) == 0
    elif number == 9:
        logits = torch.randn(2, 4, 5, requires_grad=True)
        y = torch.randint(0, 5, (2, 4))
        result = m.language_model_loss(logits, y)
        expected = sum(F.cross_entropy(logits[:, t], y[:, t]) for t in range(4)) / 4
        torch.testing.assert_close(result, expected)
        result.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    elif number == 10:
        for t in (1, 5):
            q, k, v = (torch.randn(2, 3, t, 4, requires_grad=True) for _ in range(3))
            actual = m.causal_attention(q, k, v)
            reference = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            actual.sum().backward()
            assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in (q, k, v))
            torch.testing.assert_close(actual[:, :, 0], v.detach()[:, :, 0])
        changed = v.detach().clone()
        changed[:, :, 3:] += 100
        torch.testing.assert_close(actual.detach()[:, :, :3],
                                   m.causal_attention(q.detach(), k.detach(), changed)[:, :, :3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", type=int, choices=range(1, 11))
    parser.add_argument("--solutions", action="store_true", help="Check references, not your work")
    args = parser.parse_args()
    implementation = importlib.import_module("solutions" if args.solutions else "exercises")
    torch.set_num_threads(1)
    failed = []
    for number in ([args.exercise] if args.exercise else range(1, 11)):
        try:
            check(number, implementation)
        except Exception as error:
            failed.append(number)
            print(f"Exercise {number}: {type(error).__name__}: {error}")
        else:
            print(f"Exercise {number}: passed")
    if args.solutions:
        print("Reference checks only. Your implementations are in exercises.py.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
