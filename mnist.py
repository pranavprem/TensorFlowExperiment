# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Softmax regression on MNIST.

This is the TensorFlow beginner tutorial that was modified at Silicon Valley
Code Camp 2017 (see the README). The model is unchanged: one linear layer,
softmax cross-entropy, and gradient descent. It now runs in TensorFlow 2 eager
mode, because the session API and ``tensorflow.examples`` loader it was written
against are no longer part of a supported TensorFlow install.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np


def _import_tensorflow():
    """Import TensorFlow without the CUDA and oneDNN startup banner.

    Those lines are written from C++ before ``TF_CPP_MIN_LOG_LEVEL`` takes
    effect. The GPU probe runs on the first TensorFlow op, so this also builds
    one tensor while stderr is redirected. A missing GPU is not a failure of
    this tutorial. Stderr is restored as soon as the import returns, including
    when it fails.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        import tensorflow as tf

        return tf
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        import tensorflow as tf

        tf.zeros(())
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)
    return tf


tf = _import_tensorflow()

# Keras hosts a single npz of the Yann LeCun digits. The old tutorial downloaded
# four gzip files from the CVDF mirror of yann.lecun.com, which is no longer a
# reliable default.
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"

# The camp's script changed the published tutorial (1,000 steps of 100 images)
# to a shorter, noisier run and printed test accuracy every fifth step.
STEPS = 550
BATCH_SIZE = 10
LEARNING_RATE = 0.5
LOG_EVERY = 5
VALIDATION_SIZE = 5000
IMAGE_PIXELS = 784
NUM_CLASSES = 10


class DataSet:
    """Image and label arrays with the tutorial's ``next_batch`` cursor.

    ``next_batch`` shuffles once per epoch and carries a partial batch across
    the epoch boundary, matching the 2017 loader. Pass images that are already
    flattened; ``scale_mnist_images`` does that for MNIST.
    """

    def __init__(self, images, labels, seed=None):
        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if images.ndim != 2:
            raise ValueError(f"images must be rank 2, got shape {images.shape}")
        if len(images) != len(labels):
            raise ValueError(
                f"images and labels disagree: {len(images)} vs {len(labels)}"
            )
        if len(images) == 0:
            raise ValueError("data set is empty")
        self._images = images
        self._labels = labels
        self._num_examples = int(images.shape[0])
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._rng = np.random.RandomState(seed)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next ``batch_size`` examples."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._num_examples:
            raise ValueError(
                f"batch_size {batch_size} is larger than the data set "
                f"({self._num_examples})"
            )
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle()
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest = self._num_examples - start
            images_rest = self._images[start:]
            labels_rest = self._labels[start:]
            if shuffle:
                self._shuffle()
            self._index_in_epoch = batch_size - rest
            end = self._index_in_epoch
            return (
                np.concatenate((images_rest, self._images[:end]), axis=0),
                np.concatenate((labels_rest, self._labels[:end]), axis=0),
            )
        self._index_in_epoch = start + batch_size
        return self._images[start : self._index_in_epoch], self._labels[
            start : self._index_in_epoch
        ]

    def _shuffle(self):
        perm = self._rng.permutation(self._num_examples)
        self._images = self._images[perm]
        self._labels = self._labels[perm]


class MNIST(NamedTuple):
    train: DataSet
    validation: DataSet
    test: DataSet


class TrainResult(NamedTuple):
    weights: tf.Variable
    bias: tf.Variable
    logged_accuracy: list[float]
    final_accuracy: float


def one_hot(labels, num_classes=NUM_CLASSES):
    """Encode integer class ids as float32 one-hot rows."""
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("labels are empty")
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError(f"labels must be in [0, {num_classes})")
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def scale_mnist_images(images):
    """Flatten MNIST pixels and scale them from ``[0, 255]`` to ``[0, 1]``."""
    array = np.asarray(images)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array.reshape((array.shape[0], -1))
    elif array.ndim == 3:
        array = array.reshape((array.shape[0], -1))
    elif array.ndim != 2:
        raise ValueError(f"unexpected image shape {array.shape}")
    if array.shape[1] != IMAGE_PIXELS:
        raise ValueError(f"expected {IMAGE_PIXELS} pixels, got {array.shape[1]}")
    return array.astype(np.float32) / 255.0


def datasets_from_arrays(
    x_train,
    y_train,
    x_test,
    y_test,
    validation_size=VALIDATION_SIZE,
    seed=None,
):
    """Split raw MNIST arrays the way ``read_data_sets`` did.

    The first ``validation_size`` training examples are held out, in file
    order, before any shuffle. This script never trains on them. They stay
    available as ``mnist.validation`` because the 2017 loader exposed them.
    """
    if not 0 <= validation_size <= len(x_train):
        raise ValueError(
            f"validation_size must be between 0 and {len(x_train)}, "
            f"got {validation_size}"
        )
    return MNIST(
        train=DataSet(
            scale_mnist_images(x_train[validation_size:]),
            one_hot(y_train[validation_size:]),
            seed=seed,
        ),
        validation=DataSet(
            scale_mnist_images(x_train[:validation_size]),
            one_hot(y_train[:validation_size]),
            seed=seed,
        ),
        test=DataSet(scale_mnist_images(x_test), one_hot(y_test)),
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_file(url, destination, sha256):
    """Download ``url`` to ``destination`` unless that file already matches."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == sha256:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        try:
            urllib.request.urlretrieve(url, partial)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not download {url}: {exc}") from exc
        digest = _sha256(partial)
        if digest != sha256:
            raise RuntimeError(
                f"checksum mismatch for {destination.name}: "
                f"got {digest}, expected {sha256}"
            )
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def load_mnist(data_dir, validation_size=VALIDATION_SIZE, seed=None):
    """Download MNIST if needed and return train, validation, and test sets."""
    archive = ensure_file(MNIST_URL, Path(data_dir) / "mnist.npz", MNIST_SHA256)
    with np.load(archive) as data:
        return datasets_from_arrays(
            data["x_train"],
            data["y_train"],
            data["x_test"],
            data["y_test"],
            validation_size=validation_size,
            seed=seed,
        )


def accuracy(weights, bias, images, labels):
    """Fraction of rows whose predicted class matches the one-hot label."""
    logits = tf.matmul(tf.convert_to_tensor(images, dtype=tf.float32), weights) + bias
    correct = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
    return float(tf.reduce_mean(tf.cast(correct, tf.float32)))


def gradient_step(weights, bias, images, labels, learning_rate):
    """Apply one vanilla gradient-descent update.

    The tutorial avoided ``-sum(y * log(softmax(y)))`` because that form
    overflows when a logit is very negative. Softmax cross-entropy on the raw
    logits is the stable version of the same loss. Averaging over the batch
    matches ``GradientDescentOptimizer(0.5)`` from TensorFlow 1.3.
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    with tf.GradientTape() as tape:
        logits = tf.matmul(images, weights) + bias
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
    grad_weights, grad_bias = tape.gradient(loss, [weights, bias])
    weights.assign_sub(learning_rate * grad_weights)
    bias.assign_sub(learning_rate * grad_bias)
    return float(loss)


def train(
    mnist,
    steps=STEPS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    log_every=LOG_EVERY,
):
    """Train the linear classifier and print test accuracy along the way.

    Accuracy is measured on the test set *before* the update on every
    ``log_every`` step, including step 0, which is what the camp script did.
    The final line is measured after the last update. The 2017 file moved the
    tutorial's evaluation into the loop and left the "Test trained model"
    section empty, so a finished run never reported the trained weights.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")
    if mnist.train.images.shape[1] != IMAGE_PIXELS:
        raise ValueError("training images must be flattened to 784 pixels")
    if mnist.test.labels.shape[1] != NUM_CLASSES:
        raise ValueError("test labels must be 10-way one-hot vectors")

    weights = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES], dtype=tf.float32))
    bias = tf.Variable(tf.zeros([NUM_CLASSES], dtype=tf.float32))
    logged_accuracy = []
    for step in range(steps):
        if step % log_every == 0:
            current = accuracy(weights, bias, mnist.test.images, mnist.test.labels)
            logged_accuracy.append(current)
            print(f"{current:.4f}")
        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        gradient_step(weights, bias, batch_images, batch_labels, learning_rate)

    final_accuracy = accuracy(weights, bias, mnist.test.images, mnist.test.labels)
    print(f"final accuracy: {final_accuracy:.4f}")
    return TrainResult(weights, bias, logged_accuracy, final_accuracy)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the SVCC 2017 softmax-regression MNIST classifier."
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        default="data/mnist",
        help="Directory for the MNIST archive (default: data/mnist)",
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--learning_rate",
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument("--log_every", "--log-every", type=int, default=LOG_EVERY)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for the training set (default: 0)",
    )
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("steps must be non-negative")
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be positive")
    if args.log_every <= 0:
        parser.error("log_every must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    mnist = load_mnist(args.data_dir, seed=args.seed)
    train(
        mnist,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
