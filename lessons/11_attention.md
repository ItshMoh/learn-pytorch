# 11 — Implement attention where you can see the calculation

**Outcome:** build causal multi-head attention with tensor operations and verify its behavior. Budget 3–4 hours. Labs: `labs/attention.py` and `labs/11_attention.py`.

Start with one sequence and one head. Query vectors ask which information to collect; keys are compared against queries; values contain the information to aggregate. These are learned linear projections of token representations, not fixed semantic labels assigned to individual heads.

```text
Q: [T,D]  K: [T,D]  V: [T,D]
scores = Q @ K.T / sqrt(D)    [T,T]
weights = softmax(scores)    [T,T], normalize across keys
context = weights @ V       [T,D]
```

Each output row is a weighted combination of value rows. Row i represents query position i; column j represents candidate key position j. Heads learn separate projections, allowing different information to be aggregated in parallel.

## Work a numerical example first

If all query/key scores are zero, unmasked attention weights are uniform. With three positions and a causal mask:

```text
position 0 weights: [1,   0,   0]
position 1 weights: [1/2, 1/2, 0]
position 2 weights: [1/3, 1/3, 1/3]
```

With scalar values `[2,4,8]`, context is `[2,3,14/3]`. This tiny example lets you check masking independently of projections.

For the usual variance argument, assume roughly independent, zero-mean, unit-variance query/key components. Their dot product variance grows with D. Scaling by sqrt(D) limits that growth so softmax is less likely to saturate merely because the head dimension increased. It is a useful initialization intuition, not a guarantee about arbitrary learned distributions.

## Mask scores before softmax

A causal query at position i may use j only when `j <= i`. Create a lower-triangular boolean **allowed** mask and fill disallowed scores with negative infinity. Softmax then assigns them zero probability. The diagonal remains allowed, so every row has at least one finite entry. If you mask every entry in a row to negative infinity, softmax can produce NaNs.

Multiplying scores by zero does not exclude a position: `exp(0)` is positive. Masking probabilities after softmax without renormalizing also changes the intended distribution. Read each API's boolean convention; this course uses True to mean allowed.

## Multi-head shapes

```text
x                                      [B,T,C]
combined QKV projection                [B,T,3C]
each projection after head splitting   [B,H,T,D]
scores and probabilities               [B,H,T,T]
context                                [B,H,T,D]
transpose, merge                       [B,T,C]
output projection                      [B,T,C]
```

Read each line of `CausalSelfAttention.forward`. Do not move on until you can reproduce the head split and merge from Lesson 2. The implementation intentionally omits dropout, padding masks, KV caching, and cross-attention so you can establish a correct core first.

## Verify three separate claims

1. **Mathematics:** manual results match PyTorch SDPA with dropout disabled and `is_causal=True`. SDPA is used only as a check, not inside our implementation.
2. **Causality:** changing future input tokens leaves earlier outputs unchanged. Checking only shape cannot establish this.
3. **Differentiability:** backward produces finite gradients for the learned projection parameters.

```bash
python labs/11_attention.py
python check_exercises.py --exercise 10
```

Implement exercise 10 without importing SDPA or the reference implementation. Try T=1 and verify the output equals the only value vector. Then remove the mask temporarily and see the causality check fail.

## Exit checkpoint

Write scaled dot-product attention from memory. Explain softmax's axis, the mask diagonal, scaling, head splitting, and the output projection. Naive attention stores scores growing as B*H*T*T; doubling T roughly quadruples that score storage.

[SDPA reference and mask semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). Next: [capstone](12_capstone.md).
