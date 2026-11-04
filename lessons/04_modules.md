# 4 — A model is a computation plus registered state

**Outcome:** build a model, identify its trainable parameters, and fit a known function. Budget 2–3 hours. Lab: `labs/04_modules.py`.

Our regression data obey `y=3*x-0.5`. Start the model with weight and bias at zero, train, and see whether it discovers the two numbers. This deliberately simple problem gives you a known answer: you can debug the training machinery before asking it to learn an unknown pattern.

Read `ScalarRegressor` in the lab. In `__init__`, it calls `super().__init__()` and assigns two `nn.Parameter` objects. In `forward`, it specifies `x*weight+bias`. Calling `model(x)` executes the module machinery and then forward. Call the model, rather than invoking `model.forward(x)` directly, so hooks and related module behavior remain available.

## Why registration matters

| Stored on a module | Optimizer discovers it through `parameters()`? | In `state_dict()` by default? |
|---|---|---|
| `nn.Parameter` | Yes; `requires_grad` controls differentiation | Yes |
| Child `nn.Module` | Its registered parameters recursively | Its registered state recursively |
| `register_buffer("name", tensor)` | No | Yes, for persistent buffers |
| Ordinary tensor attribute | No | No |

Moving a module with `.to(device)` moves registered parameters and buffers. Ordinary tensor attributes do not get this management automatically. A frozen parameter still appears in `parameters()`; you can construct an optimizer from only parameters whose `requires_grad` is true.

Use `nn.ModuleList` for a variable number of child layers you iterate over in forward. A regular Python list does not register its contents as child modules. `nn.Sequential` registers layers and calls them in order for you. The capstone will use both.

## Linear layers and the weight convention

`nn.Linear(4,6)` has weight `[6,4]` and bias `[6]`, for 30 parameters. It computes `x @ weight.T + bias`. It accepts `[...,4]` and returns `[...,6]`, so a `[B,T,4]` tensor becomes `[B,T,6]`. There is no need to loop over tokens.

Two linear layers without a nonlinear activation compose into another affine transformation. A ReLU or GELU between them lets the model represent more varied functions. Do not infer that a deeper network always improves learning: optimization and overfitting still matter.

## Follow the regression loop

Each iteration in `fit` does five things: clear gradients, calculate predictions, calculate MSE, backpropagate, and update with SGD. `model.parameters()` supplies references to the model's registered parameters. Create the optimizer after moving the model to its training device, and before the training loop. Recreating it every batch would discard accumulated optimizer state such as momentum or Adam moments.

```bash
python labs/04_modules.py
```

The learned weight should be close to 3 and bias close to -0.5; the loss should approach zero on this noise-free problem. Print the prediction at x=2 and compare with the known answer 5.5. This is extrapolation outside the training range; the correct linear form makes it possible here, not a general promise for neural networks.

## Practice and investigate

Implement **exercise 4**, `make_mlp`: Linear → ReLU → Linear. For input=4, hidden=5, classes=3, calculate the parameter count before checking it: `(4*5+5)+(5*3+3)=43`.

```bash
python check_exercises.py --exercise 4
```

In a scratch model, replace a registered parameter with a plain `requires_grad=True` tensor. Inspect `named_parameters()`. Predict why an optimizer built from `model.parameters()` will not train that attribute. Then replace a list of layers with `ModuleList` and inspect the difference.

## Exit checkpoint

Write a small `nn.Module` from memory. Explain parameter versus buffer and count the parameters in a two-layer network. Identify where the regression loop computes gradients and where it changes values.

[Official model tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) and [Module API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html). Next: [data](05_data.md).
