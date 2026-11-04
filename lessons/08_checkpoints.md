# 8 — Saving weights is not the same as resuming training

**Outcome:** recover identical predictions and understand the extra state needed for training. Budget 2 hours. Lab: `labs/08_checkpoints.py`.

Your model's `state_dict()` maps registered names to parameter/buffer tensors. Our classifier includes `mean` and `scale` as well as neural-network weights, so preprocessing survives saving. The architecture's Python class and configuration still need to exist when reconstructing the model.

## Two purposes, two state requirements

For inference, save model state, architecture configuration, preprocessing/tokenizer metadata, and class/token meaning. For training resumption, also save optimizer state, progress, and any scheduler/scaler state. Reproducing an exact next update can additionally require random-number-generator and sampler state.

The lab saves CPU RNG state because its next step includes dropout. It compares one fixed next batch after loading. This proves that specific CPU continuation matches. A complete multi-worker/shuffled/GPU training restart needs loader position, sampler/generator state, device RNG state, and environment assumptions too. The demonstration does not claim that broader guarantee.

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
restored = Classifier(**checkpoint["config"])
restored.load_state_dict(checkpoint["model"])
restored.eval()
with torch.no_grad():
    predictions = restored(inputs)
```

`map_location` controls tensor loading location. `weights_only=True` supports tensors and suitable simple metadata while limiting the general unpickling path; it is good practice, not permission to treat arbitrary files as trusted. Our checkpoint contains only our own tensors and basic containers.

## Why predictions can change after loading

The wrong architecture, dropout still active, different preprocessing, mismatched vocabulary ordering, or different precision can change results. Diagnose the entire input-to-output contract. Loading without an exception only tells you the state matched expected keys/shapes; it does not prove semantic correctness.

## A subtle “best weights” mistake

`best_state = model.state_dict()` holds tensors that refer to current state. Further updates can change what those references see. Serialize immediately when validation improves, or clone/deepcopy the state for an independent in-memory snapshot. Our capstone saves the final model for a fixed-step experiment; making best-validation checkpoint selection is one of your extensions.

Freezing also requires care: `requires_grad_(False)` prevents future gradient computation for that parameter, but a stale existing `.grad` may still matter to an optimizer holding it. Clear gradients and rebuild/select optimizer parameter groups as appropriate when changing what trains.

## Practice

```bash
python labs/08_checkpoints.py
```

This writes `outputs/classifier.pt` and compares evaluation predictions and the next fixed CPU training step. Rerunning replaces that lesson's output file. Read the ordering of RNG restoration in the lab: constructing a new model consumes random draws, so restoring the saved RNG before construction would be too early.

Investigate: omit optimizer-state loading and compare the next AdamW update. Then omit RNG restoration. These failures have different causes; explain both. Reuse the checkpoint after a fresh Python process as an extra exercise, reconstructing the same configuration first.

## Exit checkpoint

List what belongs in an inference artifact versus a resumable checkpoint. Explain why optimizer moments matter, why preprocessing must persist, and why a matching prediction is stronger evidence than “the file loaded.”

[Official save/load tutorial](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) and [general checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html). Next: [debugging](09_debugging.md).
