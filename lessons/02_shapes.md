# 2 — Think in shapes before writing operations

**Outcome:** perform batched matrix multiplication and split attention heads without mixing examples or token positions. Budget 2–3 hours. Lab: `labs/02_shapes.py`.

Use these names throughout the course:

| Symbol | Meaning | Example |
|---|---|---|
| B | Batch size | 2 independent sequences |
| T | Tokens per sequence | 3 positions |
| C | Model channels | 4 numbers per token |
| H | Attention heads | 2 |
| D | Channels per head, C/H | 2 |
| V | Vocabulary size | Number of token IDs |

For `x: [B,T,C]` and `w: [C,O]`, `x @ w` yields `[B,T,O]`. The multiplication contracts C and applies the same matrix at every batch/position. In contrast, `x * x` multiplies corresponding elements. `@` and `*` represent different mathematics.

## Broadcasting: useful and dangerous

Align shapes from the right. Axes are compatible if sizes match or one size is 1; missing leading dimensions behave as size 1. Hence:

```text
[2, 3, 4] +       [4] -> [2, 3, 4]   feature bias
[2, 3, 4] + [2, 1, 4] -> [2, 3, 4]   per-example feature offset
   [3, 1] -       [3] ->    [3, 3]   often an unintended pairwise difference
```

Worked bug: predictions `[[1],[2],[3]]` minus targets `[1,2,3]` produce all pairwise differences, not three matched errors. The code can run and optimize the wrong objective. For elementwise regression, assert equal shapes before computing the loss.

`keepdim=True` leaves a reduced axis at size 1. For `x: [B,T,C]`, centering each token uses `x - x.mean(dim=-1, keepdim=True)`. The mean has `[B,T,1]`, so broadcasting aligns correctly.

## Rearranging data is not renaming an axis

```python
x = torch.arange(6).reshape(2, 3)
print(x.transpose(0, 1)) # [[0,3], [1,4], [2,5]]
print(x.reshape(3, 2))   # [[0,1], [2,3], [4,5]]
```

`transpose` swaps axes; `permute` specifies any axis ordering. `reshape` groups elements into a new shape, preserving logical element order rather than swapping axis meaning.

Attention head splitting therefore has two steps:

```python
# x: [B,T,C], C=H*D
heads = x.reshape(B, T, H, D).transpose(1, 2) # [B,H,T,D]
merged = heads.transpose(1, 2).reshape(B, T, C)
```

The inverse must undo the axis swap before merging. Directly reshaping `[B,H,T,D]` into `[B,T,C]` can scramble token/head associations while producing the expected shape.

A tensor has storage and strides describing how indices address that storage. Transpose often returns a non-contiguous view. `view()` requires compatible strides; `reshape()` can copy when necessary. `.contiguous()` obtains a contiguous representation. Do not scatter it everywhere: first identify the intended layout. A view can share storage; `.clone()` makes an independent copy. `.detach()` alone does not copy storage.

## Practice

Predict all shapes in the lab, then run it. Implement **exercise 2**, `split_heads`. Use an `arange` tensor rather than random values so you can track which numbers belong to each token. Merge your heads back and compare with the original.

```bash
python labs/02_shapes.py
python check_exercises.py --exercise 2
```

Break it: try C=10 and H=3. Reject this configuration with a useful exception. Then compare `cat` (extend an existing axis) and `stack` (introduce a new axis) on two `[2,3]` tensors.

## Exit checkpoint

Explain the output shape of `[2,4,7,8] @ [2,4,8,7]`. Explain why `reshape` cannot replace `transpose`. Detect the `[B,1]` versus `[B]` loss bug without running training.

[Broadcasting rules](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) and [tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html). Next: [autograd](03_autograd.md).
