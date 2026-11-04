# 12 — Build and train your first decoder-only transformer

**Outcome:** connect all course skills in a small character language model. Budget 4–5 hours, including rebuilding components and interpreting results. Lab: `labs/12_language_model.py`.

This is a real trainable transformer with deliberately tiny scale and synthetic text. It is a course on architecture and training correctness; fluent general language or assistant behavior is not a reasonable expectation from this dataset.

## Read the architecture before running it

```text
character IDs [B,T]
    -> token embeddings + learned position embeddings [B,T,C]
    -> repeated pre-norm transformer blocks [B,T,C]
    -> final LayerNorm [B,T,C]
    -> vocabulary projection [B,T,V]
    -> next-token cross-entropy against shifted targets [B,T]
```

Each block contains:

```python
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

The attention sublayer mixes information across allowed token positions. The feed-forward sublayer transforms each position independently, using shared weights across positions. Residual additions preserve a direct route through the block and allow sublayers to learn changes to a representation.

`nn.LayerNorm(C)` normalizes the last C features for each token independently. Conceptually, compute mean and population variance over that axis, divide centered values by `sqrt(variance+epsilon)`, then apply learned scale and bias. It does not average across different examples or future positions.

Learned position embeddings let the model distinguish identical characters appearing at different positions. Without position information, attention alone cannot infer arbitrary sequence order from a bag of identical token features. Learned positions also impose the model's configured context limit. RoPE is a later alternative, not needed to establish the first working model.

## Follow the dataset construction

The course creates unique short documents from color/object combinations, shuffles them with a fixed seed, and separates train documents from validation documents **before** constructing windows. A fixed character alphabet avoids learning a tokenizer on validation text. Inputs contain 32 characters; each target is shifted by one character.

Windows use a stride of 32. They do not overlap within the inputs, though the last target of a window can be the first input of the next window. Both stay inside the same split. Joining documents permits context across the newline boundary; adding explicit document-boundary handling is a later extension.

The validation documents use held-out combinations of familiar words in the same templates. Validation therefore measures this narrow synthetic distribution, not general language understanding. There is no separate test split for this capstone, so do not present its validation result as a final unbiased benchmark after repeated tuning.

## Run a controlled first experiment

```bash
python labs/12_language_model.py --steps 300
```

This defaults to CPU and writes `outputs/tiny_lm.pt`; rerunning replaces that lesson's checkpoint. The command reports parameter count, held-out loss before/after, and perplexity. `exp(mean token CE)` is perplexity when CE uses natural logs. Compare perplexity only under compatible tokenization and evaluation data.

The printed sample can improve while still being repetitive or malformed. Use the held-out objective and behavioral checks as evidence. Exact measured values from this course's verification run are in `VERIFICATION.md`.

For an optional device experiment:

```bash
python labs/12_language_model.py --steps 300 --device auto
```

Small operations can be slower on an accelerator because overhead dominates. Measure your actual workload. The example limits CPU thread count because these tiny matrix operations often do not benefit from many threads.

## Understand generation

Training evaluates predictions at all positions using known preceding tokens. Generation takes the final position's logits, samples one character, appends it, and repeats. `generate` always uses eval mode and no-grad, and the model always applies its causal mask.

The growing prompt is cropped to the last `block_size` characters before each forward pass. This implementation recomputes the whole context and restarts learned position indices within that crop. It has no KV cache. The behavior is a simple bounded-context generator, not a demonstration of long-context inference.

## Rebuild it in stages

1. Start a scratch `MyTinyLM` with embeddings and a vocabulary projection; verify `[B,T,V]`.
2. Add one attention sublayer and prove future-token invariance.
3. Add the feed-forward sublayer, LayerNorm, and residuals.
4. Add multiple blocks with `ModuleList`.
5. Write your own loss/optimizer loop, then compare with the supplied model.
6. Save/load configuration, alphabet, and state; verify identical logits on one input.

Do not count running the supplied file as completing this rebuilding exercise.

## Required experiment report

Write `scratch/capstone_report.md` with your seed/device/version, parameter count, step budget, initial/final validation loss, a fixed-prompt sample, one failure you investigated, and one limitation. Compare a one-block model with the default two-block model using the same split and a stated budget. More layers need not win at a fixed number of steps.

Extensions, in order: select the best validation checkpoint; compare a bigram baseline; add top-k sampling; implement a cache with cached/uncached output agreement; then study RoPE and GQA. For LoRA, start with a small linear layer and inspect trainable parameter counts before connecting the idea to Excali.

## Exit checkpoint

Give a five-minute explanation from characters to loss and generated text. Write the shapes and attention equation without notes. Show evidence for causality, gradient flow, checkpoint correctness, and held-out improvement. State why this project is pretraining from scratch and your Excali work is adapting pretrained weights.

The model is an educational decoder-only design. For broader architecture context see the original [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762), which describes an encoder–decoder transformer. Next: [further PyTorch](13_further_pytorch.md).
