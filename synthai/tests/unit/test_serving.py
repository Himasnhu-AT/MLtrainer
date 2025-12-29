import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pickle
import os
import numpy as np
from synthai.src.serving.app import app, startup_event

client = TestClient(app)

@pytest.fixture
def mock_model_file(tmp_path):
    model_path = tmp_path / "model.pkl"
    model_data = {
        "model": MagicMock(),
        "metadata": {"model_type": "random_forest"}
    }
    model_data["model"].predict.return_value = np.array([0, 1])
    
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    return str(model_path)

@patch("synthai.src.serving.app.os.environ.get")
def test_health_check_no_model(mock_get):
    mock_get.return_value = None
    response = client.get("/health")
    assert response.status_code == 503

def test_health_check_with_model():
    # We need to manually trigger startup or mock the global variables
    with patch("synthai.src.serving.app.model_wrapper", MagicMock()):
        with patch("synthai.src.serving.app.model_metadata", {"model_type": "test"}):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy", "model_type": "test"}

def test_predict_endpoint():
    mock_model = MagicMock()
    mock_model.model.predict.return_value = np.array([0, 1])
    
    with patch("synthai.src.serving.app.model_wrapper", mock_model):
        response = client.post("/predict", json={"data": [{"feature1": 1}, {"feature1": 2}]})
        assert response.status_code == 200
        assert response.json()["predictions"] == [0, 1]
