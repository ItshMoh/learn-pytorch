# 7 — Evaluate the model you actually intend to evaluate

**Outcome:** measure loss and accuracy correctly and distinguish evaluation mode from gradient recording. Budget 2 hours. Lab: `labs/07_evaluation.py`.

You have two separate controls:

| Control | What it changes | What it does not do |
|---|---|---|
| `model.train()` / `model.eval()` | Behavior of mode-dependent modules | Enable/disable autograd globally |
| `torch.no_grad()` | Recording of differentiable operations in the context | Change dropout or BatchNorm mode |
| `parameter.requires_grad_(False)` | Whether a leaf needs a gradient | Change module mode or freeze unrelated state |

With ordinary parameters requiring gradients, outputs can require gradients in eval mode. During a no-grad forward in train mode, dropout can still randomly mask activations. Set both the desired mode and the desired gradient context.

## Predict four combinations

The lab uses Linear → Dropout. Predict whether outputs require gradients and whether two forwards produce the same values for each combination of training/eval and grad/no-grad. In train mode, the two outputs are overwhelmingly likely to differ here, but randomness is not a logical guarantee of inequality.

Dropout zeroes random activations during training and scales survivors so their expectation is preserved. At evaluation it acts as identity. BatchNorm uses batch statistics during training and normally uses running estimates during evaluation. LayerNorm normalizes specified final dimensions within each example and does not use BatchNorm-style running estimates. Changing batch size therefore has different implications for these layers.

`torch.inference_mode()` can reduce additional autograd overhead during pure inference. Its tensors have stricter interaction with later autograd computations; begin with `no_grad()` until you know whether outputs will re-enter a training graph.

## A metric bug that passes shape checks

Suppose the first batch has four examples with mean loss 1, and the final batch has one example with loss 4. The dataset mean is `(4*1+1*4)/5=1.6`. Averaging batch means gives `(1+4)/2=2.5`, incorrectly giving the final example four times the influence of each earlier example.

Accumulate summed unweighted losses and divide by example count. For LM training with padding ignored, use the count of valid target tokens instead. Weighted losses need their own denominator; this course's classifier checker uses ordinary unweighted CE.

Accuracy counts correct predictions divided by total examples. Track it alongside loss: a classifier can improve probabilities without changing argmax predictions. Accuracy alone also hides class imbalance and calibration problems.

## Generalization is a measurement, not a mood

Decreasing training loss with rising validation loss suggests overfitting. Both remaining high can suggest underfitting, a bad learning rate, or a broken pipeline. Investigate before assigning a label. A simple model and controlled data make these distinctions easier to test.

## Practice

```bash
python labs/07_evaluation.py
python check_exercises.py --exercise 7
```

Implement CPU `evaluate_classifier`. Leave it in eval mode, disable gradient tracking, and return Python floats. Make an empty loader report a useful error. Test a final batch of size 1. Then deliberately use unweighted batch means and see which check fails.

## Exit checkpoint

Explain why `eval()` alone can still allocate an autograd graph. Explain why no-grad alone leaves dropout active. Calculate a dataset loss from unequal batches and describe how you would select a checkpoint using validation loss.

[Module training/evaluation API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) and [no_grad API](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). Next: [checkpoints](08_checkpoints.md).
