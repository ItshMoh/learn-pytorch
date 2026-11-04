# 5 — A Dataset gives one example; a DataLoader assembles batches

**Outcome:** separate data access from training and keep preprocessing free of leakage. Budget 2 hours. Lab: `labs/05_data.py`; inspect `PointsDataset` and `make_splits` in `labs/common.py`.

Our data are two-dimensional points in three clusters. We know how they were generated, so we can reason about whether a classifier should succeed. These are synthetic teaching data, not a benchmark for a production system.

## One example first

A map-style dataset supplies `__len__` and `__getitem__`. In this course, `train[0]` returns `(features, label)` with shapes `[2]` and `[]` respectively. The label is a scalar long tensor. The DataLoader collects these pairs and normally stacks them:

```text
one example:   x [2],     y []
batch of 64:   x [64,2],  y [64]
```

`batch_size` is the number of examples in a batch. `shuffle=True` changes the order of examples each traversal. An **epoch** is one traversal of the training loader. A **step** usually means one optimizer update; with accumulation, several batches can contribute to one step.

With 480 examples and batch size 128, predict `[128,128,128,96]`. The final batch can be smaller. `drop_last=True` would discard it, which can be useful in particular training configurations but would throw away evaluation examples.

Start with `num_workers=0`, making data loading happen in the main process. Multiple workers can help expensive input pipelines but add process startup and memory overhead. On macOS, process-based loading also makes the `if __name__ == "__main__":` entry-point guard important.

## Split first, fit second

Imagine validation points happen to be much larger than training points. If you compute a mean using both sets, validation information has already changed training inputs. Keep the procedure explicit:

1. Split source examples into train, validation, and test.
2. Fit preprocessing statistics on training examples only.
3. Apply those same statistics to all splits and to future inputs.

For each feature, standardize as `(x-mean)/scale`. Our scale uses population standard deviation (`correction=0`) and clamps tiny values to `1e-6`. A constant training feature then becomes zero without dividing by zero. This is a numerical safeguard, not a guarantee that an out-of-distribution value for that feature will be well behaved.

The classifier stores the mean and scale as registered buffers. That makes inference use exactly the preprocessing learned during training, including after loading a checkpoint.

## Transforms are data functions

A transform can convert a raw record into a tensor, scale a feature, tokenize text, or augment an image. You do not need torchvision to understand this idea. Here normalization is a deterministic transform. Fit any learned transform on train only. Random augmentations usually belong to training, with deterministic evaluation transforms.

The default loader collation expects stackable shapes. Variable-length text needs a custom `collate_fn` to pad sequences and construct a validity mask; Lesson 10 covers that case.

## Practice

```bash
python labs/05_data.py
python check_exercises.py --exercise 5
```

Implement `standardize` without recomputing statistics for `other_x`. Investigate a constant feature and a validation feature far outside the training range. Change batch size and predict both the number of batches and last batch size before running.

## Exit checkpoint

Explain Dataset versus DataLoader, batch versus epoch versus step, and why normalization must use training statistics. For text documents, explain why splitting overlapping token windows randomly can leak near-duplicate context; split the source documents first.

[Official data tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [transform tutorial](https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html). Next: [classification](06_classification.md).
