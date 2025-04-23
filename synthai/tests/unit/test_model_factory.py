"""
Unit tests for the model factory module.
"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from synthai.src.models.model_factory import (
    ModelFactory, BaseModel, ClassificationModel, RegressionModel
)


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for model training."""
        # Create simple binary classification data
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        y_classification = np.array([0, 1, 0, 1, 0])
        y_regression = np.array([10.5, 20.3, 30.1, 40.8, 50.2])
        
        return {
            "X": X,
            "y_classification": y_classification,
            "y_regression": y_regression
        }
    
    def test_get_classification_model(self):
        """Test creating classification models."""
        factory = ModelFactory()
        
        # Test random forest classifier
        model = factory.get_model("random_forest", task="classification")
        assert isinstance(model, ClassificationModel)
        assert isinstance(model.model, RandomForestClassifier)
        
        # Test logistic regression
        model = factory.get_model("logistic_regression", task="classification")
        assert isinstance(model, ClassificationModel)
        assert isinstance(model.model, LogisticRegression)
    
    def test_get_regression_model(self):
        """Test creating regression models."""
        factory = ModelFactory()
        
        # Test random forest regressor
        model = factory.get_model("random_forest", task="regression")
        assert isinstance(model, RegressionModel)
        assert isinstance(model.model, RandomForestRegressor)
        
        # Test linear regression
        model = factory.get_model("linear_regression", task="regression")
        assert isinstance(model, RegressionModel)
        assert isinstance(model.model, LinearRegression)
    
    def test_infer_task_from_model_type(self):
        """Test inferring the task from model type."""
        factory = ModelFactory()
        
        # Test classification inference
        assert factory._infer_task_from_model_type("random_forest") == "classification"
        assert factory._infer_task_from_model_type("logistic_regression") == "classification"
        
        # Test regression inference
        assert factory._infer_task_from_model_type("linear_regression") == "regression"
        assert factory._infer_task_from_model_type("svr") == "regression"
    
    def test_unsupported_model_type(self):
        """Test creating a model with unsupported type."""
        factory = ModelFactory()
        
        with pytest.raises(ValueError):
            factory.get_model("unsupported_model_type")
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = ModelFactory.list_available_models()
        
        assert "classification" in models
        assert "regression" in models
        assert "random_forest" in models["classification"]
        assert "random_forest" in models["regression"]
    
    def test_classification_model_fit_predict(self, sample_data):
        """Test fitting and predicting with a classification model."""
        X = sample_data["X"]
        y = sample_data["y_classification"]
        
        # Create and fit model
        model = ClassificationModel(model_type="random_forest")
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
        
        # Test predict_proba
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y), 2)  # Binary classification
    
    def test_regression_model_fit_predict(self, sample_data):
        """Test fitting and predicting with a regression model."""
        X = sample_data["X"]
        y = sample_data["y_regression"]
        
        # Create and fit model
        model = RegressionModel(model_type="random_forest")
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_model_save_load(self, sample_data, tmpdir):
        """Test saving and loading a model."""
        X = sample_data["X"]
        y = sample_data["y_classification"]
        
        # Create and fit model
        model = ClassificationModel(model_type="random_forest")
        model.fit(X, y)
        
        # Save model
        model_path = str(tmpdir.join("test_model.pkl"))
        model.save(model_path)
        
        # Load model
        loaded_model = ClassificationModel.load(model_path)
        
        # Verify loaded model works
        predictions = loaded_model.predict(X)
        assert len(predictions) == len(y)