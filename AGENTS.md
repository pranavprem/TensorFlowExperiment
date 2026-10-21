# Agent guide

This repository is a single training script, `mnist.py`, plus tests. Read this file before changing it. [CLAUDE.md](CLAUDE.md) points here.

## What this project is

`mnist.py` is the TensorFlow beginner MNIST tutorial (softmax regression), with the edits made for Silicon Valley Code Camp 2017:

- 550 gradient steps, not 1,000
- batches of 10 images, not 100
- test-set accuracy printed every 5 steps, before that step's update

The model is still a linear classifier: 784 pixels in, 10 logits out, zero-initialized weights, mean softmax cross-entropy, vanilla gradient descent at learning rate 0.5. Those numbers are the experiment. Do not replace them with a deeper network, Adam, or a Keras `Sequential` wrapper.

A final accuracy line is printed after the last update. The 2017 file left that section empty; keep the final measurement.

## Layout

| Path | Role |
| --- | --- |
| `mnist.py` | Data loading, the training step, and the command-line entry point |
| `tests/test_mnist.py` | Unit tests. They use synthetic arrays and do not download MNIST |
| `requirements.txt` | Pinned runtime dependencies |
| `requirements-dev.txt` | Runtime pins plus pinned pytest and Ruff |
| `pyproject.toml` | Ruff and pytest configuration |
| `.github/workflows/ci.yml` | Lint, format check, and tests on Python 3.12 |

There is no package to install and no web UI.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
ruff check .
ruff format .
pytest
python mnist.py
```

`python mnist.py` downloads the MNIST npz into `data/mnist/` on first run. CI does not do that.

Useful shorter runs:

```bash
python mnist.py --steps 20 --seed 0
```

The 2017 underscore flags still work: `--data_dir`, `--batch_size`, `--learning_rate`, `--log_every`.

## Conventions

- Python 3.10 or newer. CI uses 3.12. TensorFlow 2.21 does not support 3.9.
- Format and lint with Ruff. Configuration is in `pyproject.toml`. Line length is 88.
- Keep the Apache header at the top of `mnist.py`. The training step is derived from the TensorFlow authors' tutorial. Do not add a `LICENSE` file; this refresh intentionally does not add one.
- New behavior needs a test in `tests/test_mnist.py` that does not download data or read the network.
- Prefer a plain function over a new class. `DataSet` exists to preserve `next_batch`.

## Training loop

`train` owns the loop:

1. Every `log_every` steps, measure test accuracy and print it with four decimal places. Step 0 is included, so the first number is the untrained model (about 0.1).
2. Take one gradient step on the next training batch.
3. After the loop, measure and print `final accuracy`.

`gradient_step` is vanilla SGD: `variable -= learning_rate * gradient`. The loss is `tf.nn.softmax_cross_entropy_with_logits` averaged over the batch. That is the stable form of the cross-entropy the original comments describe. Do not switch it to a softmax followed by `log`.

`DataSet.next_batch` shuffles at the start of each epoch and stitches the tail of one epoch to the head of the next. It raises if `batch_size` is larger than the set. The original loader could return a short batch in that case.

## Data

`load_mnist` downloads `mnist.npz` from the Keras storage mirror and checks its SHA-256. Images are flattened to 784 and scaled to `[0, 1]`. Labels are float32 one-hot vectors.

The first 5,000 training images, in file order, become `mnist.validation`. Nothing in `train` reads them. Leave the split in place so the loader still matches `read_data_sets` from TensorFlow 1.3.

The default `--seed` is 0. It only affects the training-set shuffle. A 2017 run was unseeded, so logged numbers will not match an old console.

## Gotchas

- Do not put a directory named `tensorflow` in the repo root. The 2017 virtualenv was checked in under that name, and Python imported it instead of the real TensorFlow package.
- Do not commit `data/`, virtualenvs, or bytecode. They are gitignored. The old virtualenv was about 400MB of Python 2.7 site-packages and a CPython binary.
- `import tensorflow` is slow. Tests pay that cost once per session. Keep tests numerical and small.
- `_import_tensorflow` briefly redirects stderr around the import and builds one tensor there. TensorFlow 2.21 prints a CUDA and oneDNN banner from C++ before `TF_CPP_MIN_LOG_LEVEL` applies, and the GPU probe (including `cuInit` failing on a machine with no GPU) happens on the first op. Leave both the redirect and the `tf.zeros(())` probe in place. Stderr is restored immediately afterward.
- Accuracy on the real test set is computed on all 10,000 images. That is cheap for this model; do not subsample it to "speed up" the log.
- Checksums live in `MNIST_SHA256`. If the upstream archive changes, update the hash on purpose. A mismatch must fail the run, not train on a partial file.
