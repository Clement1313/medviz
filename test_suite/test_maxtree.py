import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from traitement.segmentation.maxtree import *

class TestMaxtreeFonctionnels(unittest.TestCase):
    def test_build_maxtree(self):
        image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        tree, altitudes, img_gray, graph = build_maxtree(image)
        self.assertIsNotNone(tree)
        self.assertEqual(img_gray.shape, (2, 2))
        self.assertTrue(tree.num_vertices() > 0)

    def test_build_maxtree_2(self):
        image_rgb = np.ones((2, 2, 3), dtype=np.uint8) * 100
        tree, altitudes, img_gray, graph = build_maxtree(image_rgb)
        self.assertEqual(img_gray.ndim, 2)
        self.assertEqual(img_gray.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
