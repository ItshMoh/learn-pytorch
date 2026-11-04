# Reference answers — attempt before opening

Use this to correct an explanation you already wrote. After reading an answer, close it and solve a slightly changed problem. Implementations for the ten stubs are in `solutions.py`; the checks can validate them with `python check_exercises.py --solutions`.

## Exercise hints, then reasoning

| Exercise | First hint | Why the reference works |
|---|---|---|
| 1: feature means | Which axis holds examples? | Reducing axis 0 combines examples and retains one value per feature. |
| 2: split heads | Split C into H,D, then move H | Reshape alone cannot exchange tokens and heads. Check values with `arange`, not only shape. |
| 3: weight gradient | Create a fresh leaf | A detached clone isolates the caller's storage/history; MSE backward yields the scalar derivative. |
| 4: MLP | Register both linears | Sequential registers submodules, and ReLU introduces nonlinearity. Count both weights and biases. |
| 5: standardize | Fit once on train | Using training statistics for both outputs prevents the other set from influencing preprocessing. Clamping handles constant features. |
| 6: CE | Raw scores go into the loss | Stable log-softmax handles large logits and yields negative log probability of the target class. |
| 7: evaluation | Both eval and no-grad | One sets layer behavior, the other prevents graph recording. Summed losses divided by sample count handle the last batch. |
| 8: masked pooling | Broadcast a final size-1 axis | A `[B,T,1]` mask weights every channel at each valid token; the denominator counts tokens, not channels. |
| 9: token CE | Preserve vocabulary dimension | Flattening B,T produces one classification example per position with V candidate classes. |
| 10: attention | Remove future positions from score normalization | Negative infinity before softmax yields zero future probability; a first-token row contains only its own value. |

## Shape answers

For `x=[[1,2,3],[4,5,6]]` with floating point dtype:

- `x[0]` -> `[1,2,3]`, shape `[3]`.
- `x[:,1]` -> `[2,5]`, shape `[2]`.
- `x[:,1:2]` -> `[[2],[5]]`, shape `[2,1]`.
- `x.mean(0)` -> `[2.5,3.5,4.5]`.
- `x.mean(1)` -> `[2,5]`.
- `x[x>3]` -> `[4,5,6]`, shape `[3]`.
- `[2,4,7,8] @ [2,4,8,7]` -> `[2,4,7,7]`.
- `cat([a,b],dim=0)` for two `[2,3]` tensors -> `[4,3]`; `stack([a,b],dim=0)` -> `[2,2,3]`.

## Autograd answers

For `L=(2w-5)^2`, chain rule gives `dL/dw=4(2w-5)`. At w=1, gradient=-12 and an LR=0.1 update gives w=2.2, loss=0.36. LR=0.01 gives w=1.12, loss=7.6176; LR=1 gives w=13, loss=441. A sufficiently large step can increase the objective even when the gradient is correct.

For mean squared error `L=mean((wx-y)^2)`, the gradient is `mean(2(wx-y)x)`. The reference checker compares autograd with both this expression and a float64 finite-difference approximation. These are independent ways of challenging an implementation.

Accumulation adds gradients into leaves. Calling backward again on a fresh forward graph can accumulate; trying to reuse already-freed intermediates in the same graph is a different issue. Clearing `.grad` solves accumulation; recomputing the forward pass solves ordinary graph reuse between optimization steps.

## Module and data answers

Linear(4,5) has 20 weights and 5 biases. Linear(5,3) has 15 weights and 3 biases: total 43. A plain list of Linear modules does not make them registered children; ModuleList does. A buffer can move/save with a module without entering `parameters()`.

480 examples with batch size 128 produce four batches, with the last containing 96. The training mean of `[1,3]` is 2 and population standard deviation is 1. A future value 5 becomes 3 after scaling, not 0; calculating statistics on that future set would change the procedure.

With raw documents, split documents before windows. Otherwise neighboring overlapping windows can share almost all their tokens across train and validation, creating an overly favorable evaluation.

## Training and evaluation answers

Cross-entropy uses logits rather than argmax because argmax discards distribution information and is not the differentiable objective needed here. A one-third chance accuracy reference is appropriate only for this roughly balanced three-class example; another dataset may require a majority-class baseline or different metrics.

For the lab's ordinary dropout model:

| Mode | Gradient recording | Output requires grad? | Repeated outputs |
|---|---|---|---|
| Train | On | Yes | Usually different |
| Train | Off | No | Usually different |
| Eval | On | Yes | Same for this model/input |
| Eval | Off | No | Same for this model/input |

The unequal-batch loss is 1.6, not 2.5. A validation loop measures without updating weights. Its purpose is to guide model/settings selection; a separate final test estimates performance after those choices.

## Checkpoint and debugging answers

Omitting optimizer state from an AdamW resume changes moment estimates and step counters. Omitting RNG state changes dropout masks. These can each alter the next update despite identical starting weights. A full experiment restart also needs sampler/loader position and relevant environment/device state; our fixed-next-batch CPU demonstration has narrower scope.

If a gradient is `None`, ask whether the parameter participated in the differentiable computation and whether its gradient was cleared or frozen. A zero gradient differs: a path can exist while its derivative is zero. A model that overfits a tiny batch proves it can optimize that batch; it does not prove correct data splitting or generalization.

## Text, attention, and capstone answers

With fixed Q/K scores all zero, causal attention over scalar values `[2,4,8]` yields `[2,3,14/3]`. Masking to zero before softmax would leave positive probability at disallowed keys. The first output under correct causal attention is exactly the first value, because only one key is available.

IDs `[B,T]` select embeddings `[B,T,C]`; a vocabulary projection gives logits `[B,T,V]`. The class-axis convention for higher-dimensional CE differs from this output ordering, so flatten B,T or explicitly transpose to the expected class axis.

The capstone uses supervised targets derived from the text itself: next-character self-supervision. It trains its weights from random initialization. Excali starts from a pretrained model and adapts a subset of effective weights with QLoRA. Both require correct loss/data/evaluation, but their compute, initialization, and goals differ.

## Suggested oral mock

Take 15 minutes. Explain the scalar derivative, write a complete training loop, trace attention shapes, diagnose a dropout evaluation bug, and describe what belongs in a checkpoint. Give yourself credit only for answers you can defend with a small example. Use `FIELD_GUIDE.md` to locate the lesson for any missed concept.
