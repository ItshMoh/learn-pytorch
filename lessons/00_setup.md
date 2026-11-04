# 0 — Set up once, then make the computer prove it works

Your goal for this session is to create a tensor and explain which Python environment executed it. You need Python functions, loops, classes, indexing, and basic multiplication of matrices. If classes are unfamiliar, review `class`, `__init__`, and `self` before Lesson 4. You can learn the necessary calculus in Lesson 3.

## Create an isolated environment

Open a terminal in `pytorch_course`. On your Mac, Python 3.12 is available, so use it explicitly:

```bash
cd /Users/mohankumar/Downloads/code/interview_prep/pytorch_course
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.ones(2, 3))'
```

Use `python -m pip` so the installer belongs to the same Python that runs your code. Activating the environment affects this terminal; open a new terminal and activate again. The isolated temporary environment used to verify the course is separate from your learning environment.

Expected tensor:

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

If no wheel matches your Python/platform, consult the [official installation selector](https://pytorch.org/get-started/locally/). This course was written for Python 3.12 and current PyTorch 2.x. The exact environment tested is recorded in [verification](../VERIFICATION.md).

## Use CPU first

All required labs run on CPU in float32. A GPU is unnecessary for understanding the examples. The shared helper accepts `--device auto` in labs that expose it: it chooses CUDA, then Apple MPS, then CPU. Begin with CPU to establish a working baseline. Move the model and each input batch to the same device when experimenting with acceleration.

The first import can take a moment. A message about unavailable NumPy is harmless for these required labs: they do not use NumPy. If you want tensor/NumPy interoperability later, install it inside the same environment with `python -m pip install numpy`.

## Run files, not fragments of a hidden notebook state

```bash
python labs/01_tensors.py
```

Each lab is a standalone script except for explicitly imported local helpers. Re-running starts fresh. If you choose notebooks, select the `.venv` interpreter, and periodically restart the kernel and run all cells in order. A notebook can otherwise appear to work because a variable survived from an earlier experiment.

## Make a learning record

Create `scratch/notes.md` and write three columns: **prediction / observation / explanation**. For each experiment, fill the first column before running it. When the result surprises you, identify the mistaken assumption. “It works” is less useful than “reducing dimension 0 combined examples, leaving one value per feature.”

You can inspect supplied checks later:

```bash
python verify_course.py
```

This verifies the reference material. Your own learning exercises use `check_exercises.py`.

## Exit checkpoint

- Find the Python executable used by your terminal.
- Explain what a virtual environment isolates.
- Create a 2-by-3 tensor without copying the command above.
- Run a lab from a fresh terminal after activating the environment.

Continue to [Lesson 1](01_tensors.md).
