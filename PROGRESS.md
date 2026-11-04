# My PyTorch learning record

Mark a lesson only after you can reproduce its key computation without the reference. Reading and running are intermediate steps. Keep your own explanations/experiments in `scratch/`.

| Lesson | Read + ran lab | Built independently | Explained next day | Evidence / remaining question |
|---|---|---|---|---|
| 0 Setup | [ ] | [ ] | [ ] | Interpreter and first tensor |
| 1 Tensors | [ ] | [ ] | [ ] | Exercise 1 |
| 2 Shapes | [ ] | [ ] | [ ] | Exercise 2; head split/merge |
| 3 Autograd | [ ] | [ ] | [ ] | Exercise 3; derivative on paper |
| 4 Modules | [ ] | [ ] | [ ] | Exercise 4; regression from memory |
| 5 Data | [ ] | [ ] | [ ] | Exercise 5; train-only statistics |
| 6 Training | [ ] | [ ] | [ ] | Exercise 6; independent training loop |
| 7 Evaluation | [ ] | [ ] | [ ] | Exercise 7; four-mode table |
| 8 Checkpoints | [ ] | [ ] | [ ] | Load after fresh process; identical logits |
| 9 Debugging | [ ] | [ ] | [ ] | Overfit 16 examples; bug investigations |
| 10 Text | [ ] | [ ] | [ ] | Exercises 8–9; padding invariance |
| 11 Attention | [ ] | [ ] | [ ] | Exercise 10; causality demonstration |
| 12 Capstone | [ ] | [ ] | [ ] | Independent model + experiment report |

## After each study session

Date / time spent:

What I predicted:

What the experiment showed:

The assumption I corrected:

What I can now write from a blank file:

One question for the next session:

## Practice spaced across sessions

- At the beginning of each session: reproduce yesterday's key operation in 10 minutes.
- Every three sessions: answer five earlier exit questions aloud without notes.
- After Lesson 7: rebuild the classifier loop using only shape/loss specifications.
- After Lesson 11: write attention, then explain its masking with the three-value example.
- After Lesson 12: give the five-minute interview explanation and record weak points.

## Capstone report evidence

- [ ] Environment, device, seed, parameter count, and step budget recorded.
- [ ] Training/validation source split explained.
- [ ] Validation loss before/after measured.
- [ ] Fixed-prompt sample included with limitations.
- [ ] One-versus-two-block comparison uses a stated common budget.
- [ ] Future-token invariance and checkpoint roundtrip demonstrated.
- [ ] Connection to Excali: pretraining versus QLoRA adaptation explained.

Course checks passing establishes that the supplied examples work. My independent implementations and explanations establish what I have learned.
