"""Tests for the softmax-regression training loop and MNIST loader."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mnist import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOG_EVERY,
    MNIST,
    STEPS,
    DataSet,
    accuracy,
    datasets_from_arrays,
    ensure_file,
    gradient_step,
    one_hot,
    parse_args,
    train,
)


def test_parse_args_keeps_the_camp_defaults():
    args = parse_args([])
    assert args.steps == STEPS == 550
    assert args.batch_size == BATCH_SIZE == 10
    assert args.learning_rate == LEARNING_RATE == 0.5
    assert args.log_every == LOG_EVERY == 5
    assert args.seed == 0
    assert args.data_dir == "data/mnist"


def test_parse_args_accepts_the_2017_flag_spelling():
    args = parse_args(["--data_dir", "cache", "--batch_size", "16", "--steps", "3"])
    assert args.data_dir == "cache"
    assert args.batch_size == 16
    assert args.steps == 3


def test_parse_args_rejects_bad_training_values():
    with pytest.raises(SystemExit):
        parse_args(["--steps", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--batch_size", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--learning_rate", "0"])


def test_one_hot_encoding():
    encoded = one_hot([0, 2, 9])
    assert encoded.shape == (3, 10)
    assert encoded.dtype == np.float32
    assert encoded.tolist() == [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]


def test_one_hot_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        one_hot([10])


def test_datasets_hold_out_the_original_validation_split():
    x_train = np.zeros((20, 28, 28), dtype=np.uint8)
    x_train[:5] = 255
    y_train = np.zeros(20, dtype=np.uint8)
    y_train[:5] = 3
    y_train[5:] = 4
    x_test = np.full((7, 28, 28), 128, dtype=np.uint8)
    y_test = np.ones(7, dtype=np.uint8)

    mnist = datasets_from_arrays(
        x_train, y_train, x_test, y_test, validation_size=5, seed=1
    )

    assert mnist.validation.num_examples == 5
    assert mnist.train.num_examples == 15
    assert mnist.test.num_examples == 7
    assert mnist.validation.images.shape == (5, 784)
    assert np.allclose(mnist.validation.images, 1.0)
    assert np.allclose(mnist.validation.labels[:, 3], 1.0)
    assert np.allclose(mnist.train.labels[:, 4], 1.0)
    assert np.allclose(mnist.test.images, 128 / 255.0)
    assert mnist.train.images.dtype == np.float32


def test_datasets_reject_a_bad_validation_size():
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    labels = np.zeros(4, dtype=np.uint8)
    with pytest.raises(ValueError):
        datasets_from_arrays(images, labels, images, labels, validation_size=5)


def test_next_batch_walks_in_order_and_wraps_the_epoch():
    images = np.arange(6, dtype=np.float32).reshape(6, 1)
    labels = np.arange(6, dtype=np.float32).reshape(6, 1)
    data = DataSet(images, labels, seed=0)

    first_images, first_labels = data.next_batch(4, shuffle=False)
    second_images, second_labels = data.next_batch(4, shuffle=False)

    assert first_images[:, 0].tolist() == [0, 1, 2, 3]
    assert first_labels[:, 0].tolist() == [0, 1, 2, 3]
    assert second_images[:, 0].tolist() == [4, 5, 0, 1]
    assert second_labels[:, 0].tolist() == [4, 5, 0, 1]
    assert data.epochs_completed == 1


def test_next_batch_shuffle_is_deterministic_for_a_seed():
    images = np.arange(8, dtype=np.float32).reshape(8, 1)
    labels = np.zeros((8, 1), dtype=np.float32)
    first = DataSet(images, labels, seed=4).next_batch(8)[0]
    second = DataSet(images.copy(), labels.copy(), seed=4).next_batch(8)[0]
    assert first.tolist() == second.tolist()
    assert sorted(row[0] for row in first.tolist()) == list(range(8))


def test_next_batch_rejects_an_impossible_batch():
    data = DataSet(np.zeros((3, 1), np.float32), np.zeros((3, 1), np.float32))
    with pytest.raises(ValueError):
        data.next_batch(0)
    with pytest.raises(ValueError):
        data.next_batch(4)


def test_zero_weights_predict_class_zero():
    images = np.zeros((10, 784), dtype=np.float32)
    labels = np.eye(10, dtype=np.float32)
    weights = tf.Variable(tf.zeros([784, 10]))
    bias = tf.Variable(tf.zeros([10]))
    assert accuracy(weights, bias, images, labels) == pytest.approx(0.1)


def test_gradient_step_matches_softmax_regression():
    weights = tf.Variable(tf.zeros([784, 10]))
    bias = tf.Variable(tf.zeros([10]))
    images = np.zeros((1, 784), dtype=np.float32)
    images[0, 0] = 1.0
    labels = np.zeros((1, 10), dtype=np.float32)
    labels[0, 0] = 1.0

    gradient_step(weights, bias, images, labels, learning_rate=0.5)

    # Softmax of a zero logit vector is uniform. The loss gradient with respect
    # to logits is softmax - labels, and this example touches only pixel 0.
    expected_row = np.full(10, -0.05, dtype=np.float32)
    expected_row[0] = 0.45
    expected_weights = np.zeros((784, 10), dtype=np.float32)
    expected_weights[0] = expected_row
    np.testing.assert_allclose(weights.numpy(), expected_weights, atol=1e-5)
    np.testing.assert_allclose(bias.numpy(), expected_row, atol=1e-5)


def _separable_digits(n_per_class, seed):
    count = n_per_class * 10
    rng = np.random.RandomState(seed)
    images = rng.normal(0, 0.01, size=(count, 784)).astype(np.float32)
    label_ids = np.tile(np.arange(10), n_per_class)
    images[np.arange(count), label_ids] += 5.0
    order = rng.permutation(count)
    return images[order], one_hot(label_ids[order])


def test_training_learns_and_reports_a_final_accuracy(capsys):
    train_images, train_labels = _separable_digits(8, seed=0)
    test_images, test_labels = _separable_digits(4, seed=1)
    mnist = MNIST(
        train=DataSet(train_images, train_labels, seed=0),
        validation=DataSet(train_images[:10], train_labels[:10]),
        test=DataSet(test_images, test_labels),
    )

    result = train(mnist, steps=30, batch_size=10, learning_rate=0.5, log_every=5)

    assert result.logged_accuracy[0] == pytest.approx(0.1)
    assert len(result.logged_accuracy) == 6
    assert result.final_accuracy > 0.9
    assert result.final_accuracy > result.logged_accuracy[0]
    assert not np.allclose(result.weights.numpy(), 0)
    output = capsys.readouterr().out.strip().splitlines()
    assert output[-1] == f"final accuracy: {result.final_accuracy:.4f}"
    assert output[0] == f"{result.logged_accuracy[0]:.4f}"


def test_ensure_file_keeps_a_file_that_already_matches(tmp_path):
    payload = b"mnist-bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "cache" / "mnist.npz"

    first = ensure_file(source.as_uri(), destination, digest)
    source.write_bytes(b"changed-after-cache")
    second = ensure_file(source.as_uri(), destination, digest)

    assert first == second
    assert second.read_bytes() == payload


def test_ensure_file_rejects_a_bad_checksum_and_removes_the_partial(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    destination = tmp_path / "out.bin"

    with pytest.raises(RuntimeError, match="checksum"):
        ensure_file(source.as_uri(), destination, "0" * 64)

    assert not destination.exists()
    assert not Path(str(destination) + ".partial").exists()
