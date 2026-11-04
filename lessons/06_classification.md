# 6 — Own the complete training loop

**Outcome:** train a classifier and explain every line between data and an optimizer update. Budget 3 hours. Lab: `labs/06_classification.py`. The loop is deliberately visible in `labs/common.py`.

## Understand the output before the loss

For K=3 classes, the model returns `[B,3]` raw scores called **logits**. They need not sum to one. Softmax turns scores into probabilities, but argmax can select a class directly from logits because softmax preserves ordering.

For one example with logits `[2,1,0]`, probabilities are approximately `[0.665,0.245,0.090]`. If the correct class is 0, negative log-likelihood is about `-log(0.665)=0.408`. If the correct class is 2, it is about 2.408. Confidently choosing the wrong class is expensive.

For ordinary multiclass classification here:

```text
logits: [B,K], floating point, raw scores
target: [B],   torch.long, class indices 0 through K-1
loss:   [],    scalar mean cross-entropy
```

Pass raw logits to `F.cross_entropy`, which combines a stable log-softmax calculation with the target loss. Passing already-softmaxed values applies the wrong transformation. For a separate binary or multilabel task, `BCEWithLogitsLoss` is useful with float targets of matching shape; it is not interchangeable with multiclass CE.

## Annotate this loop

```python
model.train()
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
```

For every line, answer: does it change model mode, move data, clear gradients, create a computation, compute derivatives, or update parameters? `model.train()` changes mode; it does not run optimization by itself.

The lab uses AdamW. SGD updates from gradients with optional momentum. Adam tracks moving estimates of first and second gradient moments; AdamW applies decoupled weight decay. You can use both without initially deriving every detail, but know that optimizer state persists across updates. Learning rate controls update scale; a loss curve is your evidence for whether a setting works.

The example uses a validation set during development and a held-out test set for a final measurement. Do not repeatedly choose settings based on the test result printed by the demo. In your own experiments, tune using validation and inspect test after settling the settings.

## What good output means

```bash
python labs/06_classification.py
python check_exercises.py --exercise 6
```

Training loss should fall and held-out accuracy should greatly exceed the roughly one-third chance level of this balanced three-class task. Exact metrics are in `VERIFICATION.md`; different versions/devices may differ. High accuracy here proves a simple training pipeline works, not that you have solved a difficult classification problem.

## Practice and investigate

Implement `classification_loss`, then compare its value with manually selecting target entries from `log_softmax`. The checker includes large logits to catch unstable exponentiation.

Rebuild `train_epoch` in a scratch file without looking at the reference. Compare learning rates 0.0001, 0.02, and 1.0 over the same number of steps and seed. Record both train and validation loss. If a setting fails, explain whether updates were too small, unstable, or simply needed more iterations.

## Exit checkpoint

Write the loop from memory. Explain why it needs no training-time argmax, why the target is an integer index, and why optimizer state belongs outside the batch loop. Connect to Excali: token prediction uses the same classification loss at many positions.

[Official optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) and [CrossEntropyLoss API](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Next: [evaluation](07_evaluation.md).
