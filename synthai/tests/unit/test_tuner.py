import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from synthai.src.models.tuner import ModelTuner
from synthai.src.models.model_factory import BaseModel

class MockModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = MagicMock()
        self.model.get_params.return_value = {}
        self.model.set_params.return_value = self.model
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        return np.zeros(len(X))

def test_tuner_initialization():
    model = MockModel()
    param_grid = {"param1": [1, 2]}
    tuner = ModelTuner(model, param_grid)
    
    assert tuner.model == model
    assert tuner.param_grid == param_grid
    assert tuner.method == "grid"

@patch("synthai.src.models.tuner.GridSearchCV")
def test_tuner_grid_search(mock_grid_search):
    model = MockModel()
    param_grid = {"param1": [1, 2]}
    tuner = ModelTuner(model, param_grid, method="grid")
    
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    
    mock_search_instance = MagicMock()
    mock_search_instance.best_params_ = {"param1": 1}
    mock_search_instance.best_score_ = 0.9
    mock_search_instance.best_estimator_ = MagicMock()
    mock_grid_search.return_value = mock_search_instance
    
    results = tuner.tune(X, y)
    
    assert results["best_params"] == {"param1": 1}
    assert results["best_score"] == 0.9
    mock_grid_search.assert_called_once()
    mock_search_instance.fit.assert_called_once_with(X, y)

@patch("synthai.src.models.tuner.RandomizedSearchCV")
def test_tuner_random_search(mock_random_search):
    model = MockModel()
    param_grid = {"param1": [1, 2]}
    tuner = ModelTuner(model, param_grid, method="random", n_iter=5)
    
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    
    mock_search_instance = MagicMock()
    mock_search_instance.best_params_ = {"param1": 1}
    mock_search_instance.best_score_ = 0.9
    mock_search_instance.best_estimator_ = MagicMock()
    mock_random_search.return_value = mock_search_instance
    
    results = tuner.tune(X, y)
    
    assert results["best_params"] == {"param1": 1}
    assert results["best_score"] == 0.9
    mock_random_search.assert_called_once()
    mock_search_instance.fit.assert_called_once_with(X, y)
