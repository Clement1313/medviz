import unittest

import numpy as np
from traitement.segmentation.maxtree import build_maxtree


class TestMaxtreeFonctionnels(unittest.TestCase):
    def test_build_maxtree(self):
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        tree, altitudes, img_gray, graph, image_color = build_maxtree(image)
        self.assertIsNotNone(tree)
        self.assertEqual(img_gray.shape, (2, 2))
        self.assertTrue(tree.num_vertices() > 0)
        self.assertIsNone(image_color)

    def test_build_maxtree_2(self):
        image_rgb = np.ones((2, 2, 3), dtype=np.uint8) * 100
        tree, altitudes, img_gray, graph, image_color = build_maxtree(image_rgb)
        self.assertEqual(img_gray.ndim, 2)
        self.assertEqual(img_gray.shape, (2, 2))
        np.testing.assert_array_equal(image_color, image_rgb)


if __name__ == "__main__":
    unittest.main()
