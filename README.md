# TensorFlow Experiment

A one-file softmax-regression classifier for handwritten digits. Pranav Prem saved it on October 14, 2017 as a modified copy of the TensorFlow beginner tutorial, from a Silicon Valley Code Camp 2017 session. The model is the same one the tutorial taught. The camp script trained it for fewer, smaller steps and printed test accuracy while it trained.

It runs today on Python 3.12 and TensorFlow 2.21. Progress is a column of accuracies in the terminal.

## Original context

Silicon Valley Code Camp 2017 was held October 7–8. The only description in the original repository was the line in `README.md`: "MNIST Tensor Flow tutorial modifications from SVCC '17". The script that landed with that note is the [beginner MNIST tutorial](https://www.tensorflow.org/get_started/mnist/beginners) shipped with TensorFlow 1.3, with three edits:

| | Published tutorial | This script |
| --- | --- | --- |
| Steps | 1,000 | 550 |
| Batch size | 100 | 10 |
| When accuracy is printed | Once, after training | Every 5 steps, before that step's update |

So a run sees 5,500 training images instead of 100,000. The learning rate stayed 0.5. The first printed number is the untrained model, which scores about 0.10 because a tie goes to class 0.

The second commit also checked in a Python 2.7 virtualenv named `tensorflow/`, with TensorFlow 1.3.0, NumPy 1.13.3, and the packages TensorBoard needed at the time. That environment is not part of the project anymore. It was about 400MB, it cannot run on a current interpreter, and a local directory named `tensorflow` shadows the installed package.

## How it works

Each example is a 28×28 grayscale digit, flattened to 784 pixels in `[0, 1]`, with a 10-way one-hot label.

The model is a linear layer, initialized at zero:

```text
logits = images @ W + b
```

`W` is 784×10 and `b` is 10. The predicted class is the largest logit. Training minimizes the mean softmax cross-entropy of those logits. That is the numerically stable form of `-sum(label * log(softmax(logits)))`; the softmax-then-log version overflows when a logit is very negative, which is why the original tutorial called it out.

Each step is vanilla gradient descent with learning rate 0.5:

```text
W := W - 0.5 * d(loss)/d(W)
b := b - 0.5 * d(loss)/d(b)
```

The training set is shuffled once per epoch. The first 5,000 images of the official training file are held out as `mnist.validation`, in file order, before that shuffle. The tutorial's loader did the same thing. This script never trains on them and never evaluates them; test accuracy is always the full 10,000-image test set.

After the last of the 550 updates, the script prints `final accuracy`. The 2017 file had a comment for that check and no code under it, so a finished run never scored the weights it had just trained.

## Run it today

Use Python 3.10 or newer. The commands below were run on Python 3.12, which TensorFlow 2.21 supports.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python mnist.py
```

The first run downloads `mnist.npz` (about 11MB) into `data/mnist/` and checks its SHA-256. Later runs reuse the file. Pass `--data_dir` to put it somewhere else. That flag name is the one from 2017; `--data-dir` also works.

```bash
python mnist.py --steps 20 --seed 0
python mnist.py --data_dir /tmp/mnist --batch_size 10
```

`--seed` defaults to 0 so the training shuffle is repeatable. The 2017 script did not set a seed.

A full run with the camp defaults and `--seed 0` prints test accuracy every five steps, then the restored final score. The middle of the log is the same kind of line, omitted here:

```text
0.0980
0.3129
0.5767
0.4088
0.6442
...
0.8807
0.8790
final accuracy: 0.8738
```

The early jumps, including the drop from 0.58 back to 0.41, come from a batch size of 10 at learning rate 0.5. A finished run lands near 0.87. That is what 5,500 updates of this linear model produce.

## Tech stack

| | Then (October 2017) | Now |
| --- | --- | --- |
| Language | Python 2.7 | Python 3.10+ (tested on 3.12) |
| Framework | TensorFlow 1.3.0, graph and session | TensorFlow 2.21.0, eager mode |
| Arrays | NumPy 1.13.3 | NumPy, pinned via TensorFlow in `requirements.txt` |
| Data | `tensorflow.examples.tutorials.mnist` | `mnist.npz` from the Keras storage mirror |
| Tests | none | pytest 9.1.1 |
| Lint and format | none | Ruff 0.17.0 |
| CI | none | GitHub Actions, Python 3.12 |

Runtime pins are in `requirements.txt`. Development pins are in `requirements-dev.txt`.

## Project structure

```text
mnist.py                  training script and data loader
tests/test_mnist.py       synthetic tests (no download)
requirements.txt          pinned runtime dependencies
requirements-dev.txt      runtime pins plus pytest and Ruff
pyproject.toml            Ruff and pytest settings
.github/workflows/ci.yml  lint, format, and tests
AGENTS.md                 guide for changing the code
```

## Development

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format .
pytest
```

The tests build small arrays in memory. They check the one-hot encoding, the 5,000-image validation split, batching, the gradient step against the softmax-regression derivative, and that a separable toy problem actually trains. They do not contact the network.

GitHub Actions runs those commands on Ubuntu with Python 3.12. See [AGENTS.md](AGENTS.md) before changing the training loop.
