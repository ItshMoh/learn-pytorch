# Source map and further references

The course uses original synthetic examples, exercises, and explanations tailored to implementing a language model. It is not a reproduction of the FashionMNIST tutorial. The following official pages were used to check the core workflow and API behavior; links within advanced lessons also point to further reading.

## How it covers PyTorch Learn the Basics

| Official section | Course coverage | What this course adds for your goal |
|---|---|---|
| [Introduction](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) and [Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) | Setup and course map; Lessons 4–8 | A staged progression, independent exercises, and learning checkpoints |
| [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) | Lessons 1–2 | Attention-oriented axis reasoning and silent-broadcast investigations |
| [Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) | Lesson 5 | Train-only preprocessing and source-document split reasoning |
| [Transforms](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) | Lesson 5; optional image route in 13 | Transform concepts without requiring an image library |
| [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) | Lesson 4 | Parameters/buffers/ModuleList and known-answer regression |
| [Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | Lesson 3 | Hand-derived and numerical derivative checks |
| [Optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Lessons 6–7,9 | Unequal-batch metric checks and controlled debugging |
| [Save and Load](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) | Lesson 8 | Model/preprocessing roundtrip and fixed-next-step CPU continuation |

## API references used for deeper checks

- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html): compatible axes and broadcast behavior.
- [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): registered state and module mode.
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): logits, targets, and class-axis contracts.
- [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): reference computation, causal masking, functional dropout behavior.
- [AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): autocast, scaling, and clipping order.
- [Installation selector](https://pytorch.org/get-started/locally/): platform-specific installation guidance.

## When you finish

Use [Raschka's LLMs from Scratch repository](https://github.com/rasbt/LLMs-from-scratch) for a larger implementation and further training exercises. Revisit the [mini-course you originally shared](https://machinelearningmastery.com/building-transformer-models-from-scratch-with-pytorch-10-day-mini-course/) after a plain attention/GPT implementation works. Introduce its modern components one at a time and verify their shape conventions and evaluation behavior.

For paper reading, the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) describes an encoder–decoder model; our capstone implements a simpler decoder-only language model. Do not treat those architectures as identical.

The source APIs are current web references and can change. The validation report records the package version that actually executed the supplied code, which is stronger evidence than a documentation version label alone.
