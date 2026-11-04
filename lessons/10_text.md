# 10 — Text becomes IDs, then vectors, then predictions

**Outcome:** batch variable-length sequences, mask padding, and construct a next-token objective. Budget 2–3 hours. Lab: `labs/10_text.py`.

An integer ID is an address in a vocabulary, not a measurement. If “bird” is ID 2 and “stone” is ID 8, that does not mean stone is four times bird. `nn.Embedding(V,C)` stores a trainable `[V,C]` table. Passing long IDs `[B,T]` selects vectors `[B,T,C]`. These vectors acquire useful relationships through the training objective; a randomly initialized lookup table does not already contain language knowledge.

## Variable lengths require an explicit policy

Two sequences `[1,2,3]` and `[2]` cannot be stacked directly. Pad with reserved ID 0:

```text
ids:     [[1,2,3], [2,0,0]]   shape [2,3]
valid:   [[T,T,T], [T,F,F]]   shape [2,3]
```

The lab's `collate_tokens` constructs both. In a DataLoader, you can pass such a function as `collate_fn`; the function receives a list of examples and returns a batch.

`padding_idx=0` tells an embedding layer not to update that lookup row through ordinary embedding gradients. It is convenient, but does not automatically make all downstream computations padding-aware. Biases and attention can still produce nonzero values at padded positions. Decide where padding must be excluded.

## Masked pooling

For a sequence classifier, a simple baseline averages valid token vectors. Multiply embeddings by `valid.unsqueeze(-1)` so `[B,T,1]` broadcasts over C, sum over T, and divide by the number of valid tokens. Averaging over the padded length instead would make the representation depend on how much padding its batch happened to need.

Implement **exercise 8**, `masked_mean`. Our contract requires an all-padding example to return zeros: clamp the denominator to at least 1. In a real dataset you might instead reject empty examples; the key is choosing and documenting a policy.

## Next-token learning is classification at every position

Take token stream `[1,2,3,4,5]`:

```text
input x:  [1,2,3,4]
target y: [2,3,4,5]
```

The model's output at the position containing 1 should predict 2. It sees the true earlier tokens during training, commonly called teacher forcing. A causal mask prevents it from using the answer at a future input position.

For a batch, the model returns `[B,T,V]` logits and targets are `[B,T]`. To use the simple 2D CE interface, flatten B and T:

```python
loss = F.cross_entropy(logits.reshape(B*T, V), targets.reshape(B*T))
```

Do not pass `[B,T,V]` directly while assuming CE always treats the last axis as classes. Its higher-dimensional interface expects classes on axis 1. Flattening explicitly avoids that mismatch.

Implement **exercise 9**, `language_model_loss`. These exercise targets have no padding. With padded targets, separately mask loss using `ignore_index` and normalize metrics by valid targets. An ignored loss does not by itself prevent attention from reading padding; attention masking and loss masking serve different purposes.

## Practice

```bash
python labs/10_text.py
python check_exercises.py --exercise 8
python check_exercises.py --exercise 9
```

Add extra padding to the same sequences and prove masked pooling stays unchanged. Inspect the embedding gradient after repeating a token ID: its contributions accumulate in the same table row. Explain why sequence classification and next-token prediction use different target shapes.

## Exit checkpoint

Trace `[B,T] -> [B,T,C] -> [B,T,V]`. Write shifted inputs/targets for a six-token stream. Explain the three separate roles of a padding ID, an attention mask, and an ignored target in the loss.

[Embedding API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) and [padding sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html). Next: [attention](11_attention.md).
