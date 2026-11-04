# 13 — Continue learning when a task makes the next tool useful

The required lessons cover the foundations for implementing a transformer. This lesson maps the rest of PyTorch so you can recognize which topic addresses a specific problem. It is not a claim that every advanced topic can be mastered in an afternoon.

## Improve a training experiment first

**Schedulers:** keep a relatively simple model and compare a fixed learning rate with a documented schedule. Step an ordinary per-update scheduler after an optimizer update; metric-driven schedulers such as ReduceLROnPlateau have a different interface. Count optimizer updates, particularly with accumulation. Save scheduler state for a meaningful resume. Learning-rate warmup and cosine decay become useful experiments once a stable baseline exists.

**Initialization:** inspect the initial activation scale and gradients. `nn.init` lets you control initialization; Linear already has a default. Compare seeds and distributions with an explicit hypothesis, rather than applying a fashionable initialization to every layer.

**Transfer learning and freezing:** train a head while freezing a feature extractor, then unfreeze selected layers with suitable optimizer groups. Freezing gradients does not automatically freeze BatchNorm running statistics. Connect this distinction to adapter training: which weights change, which activations need backward, and which optimizer states exist?

## Use acceleration after establishing correctness

**Mixed precision:** autocast chooses lower precision for appropriate operations while leaving other operations in higher precision. CUDA float16 often needs gradient scaling. The safe sequence is autocast forward/loss, scaled backward, unscale, clip, scaler step, scaler update. BF16 has a different range/precision tradeoff and usually does not require the same scaling; actual support depends on device/operation.

```bash
python labs/13_optional_amp.py
```

This lab is specifically a CUDA float16 example. It reports a skip on CPU/MPS. The accelerator path must be validated on appropriate hardware; CPU course verification does not establish CUDA correctness or performance. See the [official AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

**Profiling:** measure data loading, host/device transfers, forward/backward, and optimizer time. CUDA operations are asynchronous, so naive wall-clock timing can measure launch overhead; use synchronization or appropriate timing tools. Start with [PyTorch benchmark tools](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) and [the profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

**Compilation:** compare eager execution with `torch.compile` only after outputs and training are correct. Include compilation warmup separately from steady-state throughput. Dynamic shapes and graph breaks affect results. [Compiler introduction](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

## Learn other model families when your project needs them

For images, learn `[B,C,H,W]`, convolution kernels/stride/padding, and torchvision transforms. For recurrent sequences, learn hidden state and sequence batching. For custom derivatives, study `autograd.Function` and gradient checks. For vector retrieval, implement contrastive loss and check both negative sampling and evaluation. These are extensions, not prerequisites for completing the decoder capstone.

**NumPy interoperability:** `torch.from_numpy` can share CPU storage with an array; `torch.tensor(array)` constructs a copy. A tensor requiring gradients usually needs detaching before `.numpy()`, and device tensors need a CPU transfer. Install NumPy only if doing this optional exercise. Test which edits propagate through shared storage instead of assuming every conversion copies.

## Scale only when one process becomes the constraint

DistributedDataParallel coordinates gradients across processes; data sharding and effective global batch size remain your responsibility. Sharded approaches can reduce per-device state storage. These require explicit process/device/sampler/checkpoint reasoning and deserve their own project. Begin with the [official distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).

Export and deployment require an input/output contract, compatible runtimes, artifact versioning, and representative latency/accuracy checks. Quantization changes numerical representation and deserves its own quality evaluation. Know these areas exist; choose them when a concrete deployment constraint justifies the investment.

## What to do next for your interviews

Prioritize rebuilding the capstone, then LoRA at the linear-layer level and contrastive training for retrieval. Those connect to Excali, ChatJarvis, and your target JDs. Consult [the field guide](../FIELD_GUIDE.md) and [progress sheet](../PROGRESS.md) to choose the next weak spot from evidence.
