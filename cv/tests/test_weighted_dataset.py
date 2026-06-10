import numpy as np

from weighted_dataset import (
    compute_class_weights,
    compute_image_weights,
    to_probabilities,
)


def test_compute_class_weights_is_inverse_frequency():
    w = compute_class_weights([10, 40, 50])  # total = 100
    assert np.allclose(w, [10.0, 2.5, 2.0])


def test_compute_class_weights_handles_zero_count():
    w = compute_class_weights([0, 100])  # zero -> treated as 1
    assert np.isfinite(w).all()


def test_compute_image_weights_aggregates_with_mean():
    cw = np.array([10.0, 2.5, 2.0])
    weights = compute_image_weights([[0, 0, 1], [2]], cw, agg=np.mean)
    assert np.allclose(weights, [np.mean([10.0, 10.0, 2.5]), 2.0])


def test_compute_image_weights_empty_image_is_neutral():
    cw = np.array([10.0, 2.5, 2.0])
    weights = compute_image_weights([[]], cw, agg=np.mean)
    assert weights[0] == 1.0


def test_to_probabilities_sums_to_one():
    p = to_probabilities([7.5, 2.0])
    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(p, [7.5 / 9.5, 2.0 / 9.5])
