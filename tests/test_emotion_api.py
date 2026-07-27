from pathlib import Path
import os

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "dummy_model.pth"
    torch.save({"state_dict": {"dummy": torch.randn(1)}}, model_path)
    monkeypatch.setenv("EMOTION_MODEL_PATH", str(model_path))

    if hasattr(main_module, "app"):
        main_module.app.state.startup_error = None
        main_module.service = None

    with TestClient(main_module.app) as test_client:
        yield test_client


def _write_test_image(path: Path):
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (70, 70), (170, 170), (255, 255, 255), -1)
    cv2.imwrite(str(path), image)


def test_health_and_predict_endpoint(client, tmp_path):
    image_path = tmp_path / "face.png"
    _write_test_image(image_path)

    main_module.service.face_detector.detect = lambda frame: {
        "face": np.zeros((224, 224, 3), dtype=np.uint8),
        "bbox": (0, 0, 224, 224),
        "status": "detected",
    }

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    with image_path.open("rb") as fh:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, fh, "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert "emotion" in payload
    assert "confidence" in payload
    assert "probabilities" in payload
