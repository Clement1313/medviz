import unittest

import numpy as np
from traitement.segmentation.predict import get_connected_component_masks, predict


class Mock:
    def predict_proba(self, attributes):
        return np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.0, 1.0], [1.0, 0.0]])


class TestPredictUnitaires(unittest.TestCase):
    def setUp(self):
        self.clf = Mock()
        self.attributes = np.array(
            [
                [4500, 0, 0, 0, 0],
                [4500, 0, 0, 0, 0],
                [4500, 0, 0, 0, 0],
                [6000, 0, 0, 0, 0],
                [1000, 0, 0, 0, 0],
            ]
        )

    def test_predict_standard_threshold(self):
        labels = predict(
            self.attributes,
            self.clf,
            threshold=0.5,
            n_pixels=1_000_000,
            min_area_frac=0.004,
            max_area_frac=0.005,
        )
        np.testing.assert_array_equal(labels, [1, 0, 1, 0, 0])

    def test_predict_high_threshold(self):
        labels = predict(
            self.attributes,
            self.clf,
            threshold=0.8,
            n_pixels=1_000_000,
            min_area_frac=0.004,
            max_area_frac=0.005,
        )
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0])

    def test_predict_area_limits(self):
        labels = predict(
            self.attributes,
            self.clf,
            threshold=0.5,
            n_pixels=1_000_000,
            min_area_frac=0,
            max_area_frac=0.009999,
        )
        np.testing.assert_array_equal(labels, [1, 0, 1, 1, 0])

    def test_get_connected_components_empty(self):
        mask = np.zeros((3, 3))
        masks_list = get_connected_component_masks(mask)
        self.assertEqual(len(masks_list), 1)
        self.assertEqual(masks_list[0][0], 0)

    def test_get_connected_components_multi(self):
        mask = np.array([[0, 1], [2, 2]])
        masks_list = get_connected_component_masks(mask)
        self.assertEqual(len(masks_list), 3)  # Labels 0, 1 et 2
        labels_found = [m[0] for m in masks_list]
        self.assertListEqual(labels_found, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
