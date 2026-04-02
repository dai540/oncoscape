from __future__ import annotations

import unittest

import numpy as np

from oncoscape.evaluation.metrics import balanced_accuracy, js_divergence, macro_f1, pearson_mean


class MetricsTest(unittest.TestCase):
    def test_macro_f1_perfect(self) -> None:
        y_true = np.asarray(["a", "b", "a", "b"])
        y_pred = np.asarray(["a", "b", "a", "b"])
        self.assertAlmostEqual(macro_f1(y_true, y_pred), 1.0)

    def test_balanced_accuracy_half(self) -> None:
        y_true = np.asarray(["a", "a", "b", "b"])
        y_pred = np.asarray(["a", "b", "b", "a"])
        self.assertAlmostEqual(balanced_accuracy(y_true, y_pred), 0.5)

    def test_pearson_mean_handles_constant(self) -> None:
        y_true = np.asarray([[1.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        y_pred = np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float32)
        score = pearson_mean(y_true, y_pred)
        self.assertTrue(np.isfinite(score))

    def test_js_divergence_zero_for_identical(self) -> None:
        y_true = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        y_pred = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        self.assertAlmostEqual(js_divergence(y_true, y_pred), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
