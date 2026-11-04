# 3 — Gradients answer “which way should this number move?”

**Outcome:** derive a gradient, check PyTorch's result, and perform one parameter update. Budget 2–3 hours. Lab: `labs/03_autograd.py`.

Suppose the model predicts `prediction = 2*w`, the target is 5, and the loss is `(2*w-5)^2`. A derivative measures how the loss changes for a small change to w:

```text
loss = (2w - 5)^2
d(loss)/dw = 2(2w - 5) * 2 = 4(2w - 5)
at w=1: loss=9, gradient=-12
```

The negative gradient says increasing w locally decreases loss. Gradient descent subtracts the gradient:

```text
new_w = w - learning_rate * gradient
      = 1 - 0.1 * (-12)
      = 2.2
new_loss = (4.4 - 5)^2 = 0.36
```

Do this arithmetic before executing the lab. You have now understood one training step.

## Separate three actions

```python
w = torch.tensor(1.0, requires_grad=True)
loss = (2*w - 5).square()  # build the forward computation
loss.backward()           # calculate gradients into .grad
with torch.no_grad():
    w -= 0.1 * w.grad     # update the parameter
```

`backward()` does not update w. The optimizer later performs the update using gradients. `no_grad()` around the update keeps optimizer arithmetic out of the next differentiation graph. Calling the entire forward pass under `no_grad()` during ordinary training would remove the path you need to differentiate.

## Why gradients accumulate

After a backward pass, `.grad` holds a result. A second backward on a fresh forward computation adds to it. That behavior enables gradient accumulation over multiple microbatches. For independent optimization steps, clear gradients with `optimizer.zero_grad(set_to_none=True)` or, for a single raw leaf, `w.grad = None`.

Most saved intermediates in a forward graph are released after backward. If you call `backward()` twice on the same loss, you will usually get an error. Recompute the forward pass for a new training step; `retain_graph=True` is rarely the right fix for a beginner's loop.

## Leaf tensors, detaching, and copying

A directly created trainable tensor such as w is a **leaf**. Its gradient is normally retained in `.grad`. Intermediate results can require gradients without keeping their `.grad`; use `.retain_grad()` only when inspecting those intermediates.

`w.detach()` returns a tensor disconnected from gradient history but sharing storage. `w.detach().clone()` also creates independent storage. `.item()` produces a Python scalar, useful for logging; using it inside a loss breaks the differentiable link through that value. Similarly, rebuilding a value with `torch.tensor(existing_tensor)` does not preserve the original graph.

## Two independent ways to verify a gradient

For `L(w)=mean((w*x-y)^2)`, derive `dL/dw = mean(2*(w*x-y)*x)`. Implement **exercise 3** using autograd and compare with that formula.

Then approximate the derivative numerically:

```text
dL/dw ≈ [L(w+epsilon) - L(w-epsilon)] / (2*epsilon)
```

Use float64 and epsilon around `1e-5` for this small smooth example. Finite differences are a diagnostic, not a replacement for backpropagation. Very small epsilon can amplify rounding error.

```bash
python labs/03_autograd.py
python check_exercises.py --exercise 3
```

Investigate: repeat the scalar update with learning rates 0.01, 0.1, and 1.0. Predict the first step and compare losses. “A gradient exists” does not guarantee the chosen step improves the objective.

## Exit checkpoint

Explain the difference between `requires_grad`, `.grad`, `backward`, `zero_grad`, and an optimizer step. Derive the MSE gradient without notes. Explain why `model.eval()` is not a replacement for disabling gradient recording; Lesson 7 will test this distinction.

[Official autograd introduction](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html). Next: [modules](04_modules.md).
