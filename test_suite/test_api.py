import unittest
import io
import importlib
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

TEST_DIR = tempfile.TemporaryDirectory()
TEST_ROOT = TEST_DIR.name

os.environ["DATABASE_URL"] = (
    f"sqlite:///{os.path.join(TEST_ROOT, 'retinascan_test.db')}"
)
os.environ["UPLOAD_DIR"] = os.path.join(TEST_ROOT, "uploads")

target_cwd = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "traitement")
)
os.chdir(target_cwd)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
TRAITEMENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "traitement")
)
sys.path.insert(0, TRAITEMENT_DIR)


def create_client():
    main = importlib.import_module("traitement.app.main")
    return TestClient(main.app)


client = create_client()


class TestApi(unittest.TestCase):
    def test_history_empty(self):
        response = client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_history_image_not_found(self):
        response = client.get("/history/unknown_id/image")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "not found")

    @patch("traitement.app.main.segment")
    def test_analyze_image(self, mock_segment):
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:50, 30:50] = True
        mock_segment.return_value = [(1, mask)]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bytes = io.BytesIO()
        from imageio import imwrite

        imwrite(img_bytes, img, format="png")
        img_bytes.seek(0)

        response = client.post(
            "/analyze", files={"file": ("test.png", img_bytes, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(len(data["results"]), 1)
        self.assertIn(data["results"][0]["severity"], ["faible", "modéré", "élevé"])
        self.assertIn("diagnosis", data)


if __name__ == "__main__":
    unittest.main()
