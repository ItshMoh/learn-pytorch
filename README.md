# PyTorch for building your first transformer

A practical course for pytorch ,preparing for AI engineer interviews. Your destination is being able to write, train, inspect, and debug a small language model, then explain its design without relying on a framework wrapper.

This is a complete course in the PyTorch foundations needed for that destination. PyTorch also contains specialist areas such as distributed training, custom kernels, and deployment; the final lesson gives you a route into those after the foundations. You do not need to memorize the whole API.

## Start here

Read [setup](lessons/00_setup.md), then [Lesson 1](lessons/01_tensors.md). Work in order. The explanations are the course; the Python files are experiments to run alongside it. All required datasets are generated locally. Only PyTorch is required, and CPU is the default.

From this folder, after setup:

```bash
python labs/01_tensors.py
python check_exercises.py --exercise 1
```

The first command works immediately. The second initially reports an unfinished exercise: implement the corresponding function in `exercises.py`. Reference implementations live separately in `solutions.py`.

## How to use each lesson

1. **Predict:** write the output, tensor shapes, or gradient before executing the example.
2. **Run and explain:** run the lab; explain each unfamiliar line in your own words.
3. **Rebuild:** implement the exercise without copying the reference. Spend 15–25 minutes before opening a hint or solution.
4. **Investigate:** change one thing, predict what will happen, then measure it. Record the result even when your prediction was wrong.
5. **Retrieve:** the next day, reproduce the key operation from a blank file. Passing a checker once is insufficient if you cannot explain it.

Keep experiments in your own `scratch/` folder. Every lesson has an explicit exit checkpoint; repeat the relevant lab if you cannot pass it. Reading an answer is feedback, not evidence that you learned it.

## Your path

Budget roughly **30–40 hours**, plus extra practice where needed. These are planning estimates, not deadlines. At 2 hours per session, use 15–20 sessions; do not compress unfamiliar tensor algebra into a single evening.

| Lesson | What you will build | Suggested time |
|---|---|---|
| [0 — Setup](lessons/00_setup.md) | A working environment and first tensor | 30–45 min |
| [1 — Tensors](lessons/01_tensors.md) | A batch of numerical examples | 1.5–2 h |
| [2 — Shape reasoning](lessons/02_shapes.md) | Batched projections and head splitting | 2–3 h |
| [3 — Autograd](lessons/03_autograd.md) | A derivative check and a manual update | 2–3 h |
| [4 — Models and parameters](lessons/04_modules.md) | Linear regression with `nn.Module` | 2–3 h |
| [5 — Data](lessons/05_data.md) | Dataset, DataLoader, and training-only preprocessing | 2 h |
| [6 — Classification](lessons/06_classification.md) | A complete classifier training loop | 3 h |
| [7 — Evaluation](lessons/07_evaluation.md) | Correct metrics and controlled dropout experiments | 2 h |
| [8 — Checkpoints](lessons/08_checkpoints.md) | Save/load and a resumed optimizer step | 2 h |
| [9 — Debugging training](lessons/09_debugging.md) | Gradient checks and accumulation | 2–3 h |
| [10 — Text tensors](lessons/10_text.md) | Padding, embeddings, and shifted targets | 2–3 h |
| [11 — Attention](lessons/11_attention.md) | Causal multi-head attention using tensor operations | 3–4 h |
| [12 — Language-model capstone](lessons/12_capstone.md) | A small trained decoder-only transformer | 4–5 h |
| [13 — Further PyTorch](lessons/13_further_pytorch.md) | A map of the next skills, with an optional AMP lab | Optional |

This supplements your interview study plan. Your existing Day 5 GPT task becomes the final outcome of these foundations. Keep time for Excali project defence, DSA, retrieval, and mocks if interviews are close.

## What is included

- `lessons/`: explanations, worked examples, investigations, and interview checkpoints.
- `labs/01_...py` through `labs/12_...py`: runnable examples; read the lab identified in each lesson.
- `labs/common.py`: small shared classifier/data helpers, introduced in Lessons 5–6.
- `exercises.py`: 10 functions for you to implement.
- `check_exercises.py`: feedback on those implementations, including boundary cases.
- `solutions.py` and [answers](ANSWERS.md): code references and reasoning. Open after attempting.
- [field guide](FIELD_GUIDE.md): shape contracts, debugging order, and interview drills.
- [progress sheet](PROGRESS.md): evidence of what you can do independently.
- `verify_course.py`: verifies the supplied reference labs. Its passing does **not** mean you completed the exercises.
- [sources](SOURCES.md): official documentation reviewed, including how this course maps to Learn the Basics.

## Your first session

Finish setup, spend 20 minutes predicting the examples in Lesson 1, and run `labs/01_tensors.py`. Implement exercise 1, then answer the exit questions aloud. Stop there for the first session if the concepts are new. Returning tomorrow and reproducing the work is part of the course.

## Completion standard

You are ready to continue into a larger transformer implementation when you can:

- annotate every shape from token IDs to logits;
- write a training and validation loop without a template;
- distinguish gradients, parameters, optimizer state, and buffers;
- isolate a failure with a tiny-batch experiment;
- implement masked attention without `nn.MultiheadAttention` or SDPA;
- show a checkpoint produces the same evaluation predictions after loading;
- explain what the toy capstone proves and what its synthetic data cannot establish.

All examples and exercises here are written for this course. The official documentation remains the authority for API behavior. Reviewed 2 November 2026; see `VERIFICATION.md` for the environment and checks actually executed.
