import unittest
import io
import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys
import os
target_cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'traitement'))
os.chdir(target_cwd)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
TRAITEMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'traitement'))
sys.path.insert(0, TRAITEMENT_DIR)
import traitement.app.main as main
client = TestClient(main.app)

def test_history_empty():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_history_image_not_found():
    response = client.get("/history/unknown_id/image")
    assert response.status_code == 200
    assert response.json()["error"] == "not found"



@patch("traitement.app.main.segment")
def test_analyze_image(mock_segment):
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:50, 30:50] = True
    mock_segment.return_value = [(1, mask)]
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_bytes = io.BytesIO()
    from imageio import imwrite
    imwrite(img_bytes, img, format="png")
    img_bytes.seek(0)

    response = client.post(
        "/analyze",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["severity"] in ["faible", "modéré", "élevé"]