# 1 — A tensor is numerical data with a contract

**Outcome:** construct, inspect, index, and reduce a batch. Read `labs/01_tensors.py`. Budget 1.5–2 hours including practice.

Imagine three measurements for each of two examples:

```python
import torch
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
```

`x.shape` is `[2, 3]`. We assign meaning: axis 0 is **examples**, axis 1 is **features**. PyTorch stores dimensions; your program gives them meaning. Another tensor of the same shape could mean something entirely different.

Five questions describe a tensor well: **What are its axes? What is its shape? Which dtype? Which device? Does it require gradients?** Ask these when reading unfamiliar training code.

## Predict the result

Before running anything, write the values and shapes of:

```python
x[0]
x[:, 1]
x[:, 1:2]
x.mean(dim=0)
x.mean(dim=1)
x[x > 3]
```

Indexing with `1` removes that axis. Slicing `1:2` keeps an axis of length 1. Thus `x[:, 1]` has shape `[2]`, while `x[:, 1:2]` has `[2, 1]`. This small difference can later cause a large training bug.

For reductions, read `dim=0` as “combine values along axis 0.” Combining the two rows leaves the three feature means: `[2.5, 3.5, 4.5]`. Combining axis 1 leaves two example means: `[2, 5]`. A boolean selection such as `x[x > 3]` collects selected elements into a one-dimensional result; it does not preserve the original grid.

## Dtype is part of correctness

```python
features = torch.tensor([1.0, 2.0])   # float32 under the usual default
token_ids = torch.tensor([1, 2])       # int64 / torch.long
allowed = torch.tensor([True, False]) # bool
```

Floating point supports the continuous calculations used to train weights. Class indices and token IDs identify entries, so we use integer tensors. Boolean masks select positions. Turning token IDs into floats does not make them better embeddings: the model must perform an embedding lookup.

`converted = features.to(dtype=torch.float64)` returns a tensor with the requested dtype. Likewise, assign the result of moving a tensor: `features = features.to(device)`. A tensor operation may return the same object when no conversion is needed; do not assume `.to()` always creates independent storage.

Useful constructors: `zeros`, `ones`, `arange`, `randn`, `zeros_like`. `randn` produces standard-normal samples. `empty` allocates uninitialized storage; never interpret its initial contents as data.

## Run, then investigate

```bash
python labs/01_tensors.py
```

Change the examples from two rows to four. Which reductions change shape? Then create integer features and try `mean()`. Explain the dtype problem rather than blindly adding `.float()` everywhere.

Implement **exercise 1**, `feature_means`, in `exercises.py`:

```bash
python check_exercises.py --exercise 1
```

**Hint:** the output must retain features and combine examples. Compare the result with an explicit loop over columns, then remove the loop.

## Exit checkpoint

Without running code, explain the shape of `x[:, 0]`, the difference between `sum` and `mean`, and why `[B]` is different from `[B, 1]`. Construct a batch of five examples with four features and compute one mean per feature. Interview prompt: “Why is a token-ID tensor typically integer-valued?”

[Official tensor reference tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html). Next: [shape reasoning](02_shapes.md).
