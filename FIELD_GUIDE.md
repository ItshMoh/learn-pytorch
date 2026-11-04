# PyTorch field guide

Use this after attempting a problem, or during revision. The lessons teach the reasoning; this page helps locate it.

## The recurring contracts

| Task | Prediction | Target | Loss |
|---|---|---|---|
| Scalar regression per example | `[B]` float | `[B]` float | MSE |
| Multiclass classification | `[B,K]` float logits | `[B]` long class indices | CE |
| Independent binary/multilabel labels | `[B,K]` float logits | `[B,K]` float values | BCE with logits |
| Next-token modeling | `[B,T,V]` float logits | `[B,T]` long IDs | Flatten B,T before CE |

These are the course conventions; APIs also support other forms. For elementwise regression, assert matching shapes before loss. For class indices, check `0 <= target < classes`, with any explicit ignored target handled separately.

## Training in seven lines

```python
model.train()
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x.to(device))
    loss = loss_fn(logits, y.to(device))
    loss.backward()
    optimizer.step()
```

Move the model before constructing its optimizer. If clipping, clip after backward and before step. If using scaled gradients, unscale before clipping. Clear gradients once per optimizer update; intentional microbatch accumulation has different placement.

For evaluation, set `model.eval()`, enter `torch.no_grad()`, compute summed loss/correct counts, and divide by the right denominator. Do not call backward or step. Remember to return to train mode before subsequent training.

## Ten distinctions to remember

| Pair | Difference |
|---|---|
| `*` / `@` | Elementwise multiplication / matrix multiplication |
| `reshape` / `transpose` | Regroup elements / swap axes |
| `cat` / `stack` | Extend an axis / introduce an axis |
| `.detach()` / `.clone()` | Disconnect gradient history / copy storage; clone alone can remain differentiable |
| `.item()` / tensor | Python scalar for logging / numerical object that may retain gradient history |
| `backward()` / `step()` | Compute gradients / change parameters using them |
| Parameter / buffer | Optimizer-discoverable model parameter / registered non-parameter state |
| `eval()` / `no_grad()` | Module behavior / autograd recording |
| Train loss / validation loss | Objective on optimized examples / objective on held-out examples |
| Weights / training checkpoint | Learned model values / values plus state needed to continue optimization |

## A compact glossary

**Tensor:** numerical values with shape/dtype/device. **Axis:** one dimension with meaning assigned by your program. **Broadcasting:** combining compatible shapes by matching from the right. **Logit:** raw class score. **Parameter:** registered value the model may learn. **Gradient:** local derivative of the objective with respect to a value. **Autograd graph:** differentiable operation history needed for backward. **Leaf:** tensor created directly rather than as the result of another tracked operation. **Batch:** group of examples processed together. **Epoch:** traversal of a dataset. **Optimizer state:** additional update history such as momentum. **Buffer:** registered tensor state outside parameters. **Embedding:** learnable lookup table. **Causal mask:** rule blocking future keys. **Checkpoint:** serialized state with an explicit restoration purpose.

## Attention shape rehearsal

Starting with `[B,T,C]`, project to Q/K/V and split each to `[B,H,T,D]`. Scores have `[B,H,T,T]`. Mask future keys, softmax over the last axis, multiply values, merge back to `[B,T,C]`, and project. The first row can attend only to its first value. A future-token intervention should not change any earlier output.

## Debugging order

Inspect examples → assert contracts → check gradient path → verify parameter changes → overfit a tiny batch → restore full data → examine validation. Do not skip straight to increasing compute. For reproducibility, log seed, split, model configuration, preprocessing, versions, and device; a seed alone does not promise exact cross-device equivalence.

## Interview drills linked to your goals

| Prompt | Lesson to revisit |
|---|---|
| Explain an MSE derivative and implement one update | 3 |
| Why does this trainable tensor not appear in the optimizer? | 4 |
| Why can standardizing before splitting leak information? | 5 |
| Why should CE receive logits? | 6 |
| Why is validation random despite using no-grad? | 7 |
| Why did training change after loading identical weights? | 8 |
| How would you investigate non-learning or NaNs? | 9 |
| Why are LM targets shifted and how do we batch them? | 10 |
| Implement attention and prove it is causal | 11 |
| Explain pretraining versus your Excali QLoRA work | 12–13 |

## A starting memory estimate

Parameter storage alone is `parameter_count * bytes_per_parameter`. Actual training also holds gradients, optimizer states, activations, and temporary workspaces. Attention score storage grows quadratically with sequence length in the manual implementation. Reducing parameter count does not guarantee fitting a long sequence into memory.

For faster experiments, first reduce unnecessary work or model/context/batch size. Accumulation trades more microbatch computations for lower per-forward activation memory. Activation checkpointing recomputes activations to save memory. Mixed precision changes numerical representation. These tools solve different constraints and deserve measurement.
