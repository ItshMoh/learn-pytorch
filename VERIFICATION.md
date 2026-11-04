# Course verification

The supplied reference material was executed with **Python 3.12.13, PyTorch 2.15.0, and CPU float32** in an isolated temporary environment. No external training datasets or pretrained weights were downloaded. Installing PyTorch itself requires downloading its package and dependencies.

## What ran

```bash
python check_exercises.py --solutions
python verify_course.py
```

The verification script ran all 12 core standalone lab scripts, executed the optional AMP script's CPU skip path, checked all 10 reference exercise implementations, and checked 36 local Markdown links. Temporary checkpoints were removed automatically after testing. Your exercise stubs remain unimplemented so you can do the learning work.

The environment emitted a warning about NumPy being unavailable. These required labs do not use NumPy and all checks completed successfully. See setup if you want the optional NumPy bridge exercise.

## Measured results

| Check | Observed result |
|---|---|
| Known-function regression | Final MSE about `1.92e-12`; weight/bias matched the known solution within checked tolerance |
| Three-class classifier | Training loss `0.82186` after first epoch to `0.00947` after last; held-out test accuracy `0.99167` |
| Tiny fixed batch | Loss about `3.06e-5` after 200 updates with dropout disabled |
| Classifier checkpoint | Identical evaluation logits after loading; identical next fixed CPU optimizer step with optimizer/RNG restoration |
| Manual attention | Matched SDPA reference; masked probabilities, future-token invariance, and finite gradients passed |
| Unequal microbatches | Accumulated gradients matched one full-batch calculation |
| Language model | 28,576 parameters, 300 updates; validation CE `3.60308` before training to `1.87735` after |
| Language-model perplexity | About `6.536` on the synthetic validation distribution |
| Language-model checkpoint | Identical logits after loading configuration/alphabet/state |
| Full-model causality | Changing future tokens left earlier logits unchanged |
| Generation beyond one context window | Continued successfully while cropping to the configured context length |

One sample from the verification run:

```text
the black leaf.
the red book is here. i box.
```

It demonstrates learned local patterns and remaining errors. It does not establish broad language ability. The classifier and language-model datasets are deliberately simple synthetic examples; their metrics are teaching evidence, not production benchmarks.

## Limits of this verification

- CUDA float16 AMP was **not** executed; its optional script reported a skip. MPS/CUDA training and performance were not verified.
- The exact checkpoint-continuation comparison is a fixed-next-batch CPU experiment. Full shuffled, multi-worker, accelerator training resumption needs additional state handling.
- Exact metrics can vary across versions/devices. The checks target the intended behavior and meaningful loss improvement rather than requiring these exact decimal values.
- Running the supplied course does not complete your personal exercises. Independent reconstruction and oral explanations remain part of the completion standard.
