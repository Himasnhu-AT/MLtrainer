import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from synthai.src.models.explainer import ModelExplainer
from synthai.src.models.model_factory import BaseModel

class MockModel(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = MagicMock()
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        return np.zeros(len(X))

@patch("synthai.src.models.explainer.shap")
def test_explainer_initialization(mock_shap):
    model = MockModel()
    X_train = np.array([[1], [2]])
    explainer = ModelExplainer(model, X_train)
    
    assert explainer.model == model
    np.testing.assert_array_equal(explainer.X_train, X_train)

@patch("synthai.src.models.explainer.shap")
def test_explainer_explain(mock_shap):
    model = MockModel()
    # Mock feature_importances_ to trigger TreeExplainer
    model.model.feature_importances_ = np.array([0.5])
    
    X_train = np.array([[1], [2]])
    explainer = ModelExplainer(model, X_train)
    
    X_test = np.array([[3], [4]])
    
    mock_tree_explainer = MagicMock()
    mock_shap.TreeExplainer.return_value = mock_tree_explainer
    mock_tree_explainer.shap_values.return_value = np.array([[0.1], [0.2]])
    
    explainer.explain(X_test)
    
    mock_shap.TreeExplainer.assert_called_once()
    mock_tree_explainer.shap_values.assert_called_once()
    assert explainer.shap_values is not None

@patch("synthai.src.models.explainer.shap")
@patch("synthai.src.models.explainer.plt")
def test_explainer_save_plots(mock_plt, mock_shap, tmp_path):
    model = MockModel()
    X_train = np.array([[1], [2]])
    explainer = ModelExplainer(model, X_train)
    explainer.shap_values = np.array([[0.1], [0.2]])
    
    output_dir = str(tmp_path)
    saved_plots = explainer.save_plots(output_dir)
    
    assert len(saved_plots) == 2
    # Since plt is mocked, files won't be created. Check if savefig was called.
    assert mock_plt.savefig.call_count == 2
    mock_shap.summary_plot.assert_called()
