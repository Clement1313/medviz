import unittest

import numpy as np
from traitement.segmentation.evaluation import (
    compute_iou,
    confusionCounts,
    false_negative_rate,
    false_positive_rate,
    sensitivity,
    specificity,
    weighted_error_rate,
)


class TestEvaluationUnitaires(unittest.TestCase):
    def test_iou_perfect_match(self):
        mask = np.ones((5, 5))
        self.assertEqual(compute_iou(mask, mask), 1.0)

    def test_iou_no_match(self):
        mask1 = np.ones((5, 5))
        mask2 = np.zeros((5, 5))
        self.assertEqual(compute_iou(mask1, mask2), 0.0)

    def test_iou_partial_match(self):
        mask_gt = np.array([[1, 1], [0, 0]])
        mask_pred = np.array([[1, 0], [1, 0]])
        self.assertAlmostEqual(compute_iou(mask_gt, mask_pred), 1 / 3)

    def test_iou_both_empty(self):
        mask = np.zeros((3, 3))
        self.assertEqual(compute_iou(mask, mask), 1.0)

    def test_metrics_sensitivity(self):
        self.assertEqual(sensitivity(tp=5, fn=5), 0.5)
        self.assertEqual(sensitivity(tp=10, fn=0), 1.0)

    def test_metrics_specificity(self):
        self.assertEqual(specificity(tn=8, fp=2), 0.8)
        self.assertEqual(specificity(tn=10, fp=0), 1.0)

    def test_metrics_fpr_fnr(self):
        self.assertEqual(false_positive_rate(fp=2, tn=8), 0.2)
        self.assertEqual(false_negative_rate(fn=3, tp=7), 0.3)

    def test_weighted_error_rate(self):
        self.assertAlmostEqual(weighted_error_rate(0.1, 0.2, 10), 2.1 / 11)


class TestEvaluationFonctionnels(unittest.TestCase):
    def test_confusion_counts_all_tp(self):
        mask_gt = [np.ones((2, 2))]
        mask_pred = [np.ones((2, 2))]
        tp, fp, fn, tn = confusionCounts(mask_gt, mask_pred)
        self.assertEqual((tp, fp, fn, tn), (1, 0, 0, 0))

    def test_confusion_counts_all_tn(self):
        mask_gt = [np.zeros((2, 2))]
        mask_pred = [np.zeros((2, 2))]
        tp, fp, fn, tn = confusionCounts(mask_gt, mask_pred)
        self.assertEqual((tp, fp, fn, tn), (0, 0, 0, 1))

    def test_confusion_counts_fn(self):
        mask_gt = [np.ones((2, 2))]
        mask_pred = [np.zeros((2, 2))]
        tp, fp, fn, tn = confusionCounts(mask_gt, mask_pred)
        self.assertEqual((tp, fp, fn, tn), (0, 0, 1, 0))

    def test_confusion_counts_fp(self):
        mask_gt = [np.zeros((2, 2))]
        mask_pred = [np.ones((2, 2))]
        tp, fp, fn, tn = confusionCounts(mask_gt, mask_pred)
        self.assertEqual((tp, fp, fn, tn), (0, 1, 0, 0))


if __name__ == "__main__":
    unittest.main()
