# 9 — Debug a failed experiment by reducing uncertainty

**Outcome:** locate broken training with small, controlled experiments. Budget 2–3 hours. Lab: `labs/09_debugging.py`.

When loss does not fall, do not immediately add layers or train longer. First establish that the program represents the intended task. Work in this order:

1. Inspect a few raw examples and labels; verify the label mapping.
2. Assert shape, dtype, device, and finite values before the loss.
3. Check that loss requires gradients and expected parameters receive finite `.grad` values.
4. Verify that optimizer steps actually change those parameters.
5. Train repeatedly on one tiny fixed batch, reducing regularization if necessary.
6. Only after that succeeds, restore the full data pipeline and investigate generalization.

Overfitting a tiny batch is a **sanity check**. It can expose optimization or labeling problems; succeeding does not prove that the model generalizes. Memorizing shuffled labels on a tiny set is possible too.

## Common symptoms and hypotheses

| Symptom | Check first |
|---|---|
| Loss has no gradient history | `.item()`, detach, no-grad around forward, tensor reconstruction |
| Parameter gradients are `None` | Unused layer, unregistered tensors, disconnected computation |
| Loss is finite then becomes NaN | Invalid inputs, zero division, extreme updates, unstable math |
| Training works but validation changes randomly | Dropout active, random eval transforms, changing data |
| GPU out of memory | Batch/context size; retained graphs or outputs; model/optimizer memory |
| All predictions choose one class | Target mapping, imbalance, loss setup, optimization |

Print useful summaries such as gradient norms rather than pages of tensors. Append `loss.item()` to histories instead of retaining a loss graph. If you keep predictions for later analysis, detach and move them to CPU when appropriate.

## Gradient accumulation has a denominator

Seven examples split into microbatches of 3,3,1 do not deserve equal weights per microbatch. If each microbatch loss is a mean over its examples, accumulate `loss * microbatch_size / total_examples`, then step once after all microbatches. The lab compares these gradients with one full batch.

That equivalence assumes the same per-example computation and objective. BatchNorm statistics and stochastic layers can make microbatch computation differ from a single full batch. Accumulation is not an unconditional promise of identical training.

Gradient clipping limits the combined gradient norm before an update. It does not fix incorrect labels or guarantee stable training. `clip_grad_norm_` returns the norm **before** clipping. When using gradient scaling, unscale before clipping; the optional lesson shows the ordering.

## Practice

```bash
python labs/09_debugging.py
```

Create a scratch copy of the classifier loop with one bug at a time: forget to clear gradients; detach logits; recreate the optimizer per batch; mismatch `[B]` with `[B,1]` for MSE. Predict the symptom and identify the smallest diagnostic that confirms the cause.

Then make a 16-example training subset and attempt to overfit it. Record settings, initial/final loss, and prediction accuracy. This experiment is required evidence in the progress sheet even though it is not an additional stub function.

## Exit checkpoint

Give a two-minute debugging plan for “my model is not learning.” Explain why blindly increasing compute is premature. Demonstrate that gradients exist and parameters change, and explain the unequal-microbatch calculation.

[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) and [gradient clipping API](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html). Next: [text tensors](10_text.md).
