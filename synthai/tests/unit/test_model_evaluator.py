"""
Unit tests for the model evaluator module.
"""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from synthai.src.models.evaluator import ModelEvaluator
from synthai.src.models.model_factory import ClassificationModel, RegressionModel


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    @pytest.fixture
    def classification_data(self):
        """Fixture providing classification data."""
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        y = np.array([0, 1, 0, 1, 0])
        
        # Train a simple model
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X, y)
        
        # Wrap in our model class
        wrapped_model = ClassificationModel(model_type="random_forest")
        wrapped_model.model = model
        
        return {
            "X": X,
            "y": y,
            "model": wrapped_model
        }
    
    @pytest.fixture
    def regression_data(self):
        """Fixture providing regression data."""
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        y = np.array([10.5, 20.3, 30.1, 40.8, 50.2])
        
        # Train a simple model
        model = LinearRegression()
        model.fit(X, y)
        
        # Wrap in our model class
        wrapped_model = RegressionModel(model_type="linear_regression")
        wrapped_model.model = model
        
        return {
            "X": X,
            "y": y,
            "model": wrapped_model
        }
    
    def test_evaluator_init(self):
        """Test evaluator initialization with different metrics."""
        # Default metrics
        evaluator = ModelEvaluator()
        assert evaluator.metrics == ["accuracy"]
        
        # Custom metrics
        evaluator = ModelEvaluator(metrics=["precision", "recall", "f1"])
        assert evaluator.metrics == ["precision", "recall", "f1"]
    
    def test_is_classification_task(self):
        """Test detection of classification vs regression tasks."""
        evaluator = ModelEvaluator()
        
        # Classification data
        assert evaluator._is_classification_task(np.array([0, 1, 0, 1])) is True
        
        # Regression data
        assert evaluator._is_classification_task(np.array([1.5, 2.3, 3.7, 4.1])) is False
    
    def test_evaluate_classification(self, classification_data):
        """Test evaluation with classification metrics."""
        X = classification_data["X"]
        y = classification_data["y"]
        model = classification_data["model"]
        
        # Evaluate with default metric (accuracy)
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(model, X, y)
        
        assert "accuracy" in results
        assert 0 <= results["accuracy"] <= 1
        
        # Evaluate with multiple metrics
        evaluator = ModelEvaluator(metrics=["accuracy", "precision", "recall", "f1"])
        results = evaluator.evaluate(model, X, y)
        
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        
        # Compare with sklearn directly
        y_pred = model.predict(X)
        expected_accuracy = accuracy_score(y, y_pred)
        expected_f1 = f1_score(y, y_pred, average='binary')
        
        assert results["accuracy"] == pytest.approx(expected_accuracy)
        assert results["f1"] == pytest.approx(expected_f1)
    
    def test_evaluate_regression(self, regression_data):
        """Test evaluation with regression metrics."""
        X = regression_data["X"]
        y = regression_data["y"]
        model = regression_data["model"]
        
        # Evaluate with regression metrics
        evaluator = ModelEvaluator(metrics=["rmse", "mse", "mae", "r2"])
        results = evaluator.evaluate(model, X, y)
        
        assert "rmse" in results
        assert "mse" in results
        assert "mae" in results
        assert "r2" in results
        
        # Compare with sklearn directly
        y_pred = model.predict(X)
        expected_mse = mean_squared_error(y, y_pred)
        expected_rmse = np.sqrt(expected_mse)
        expected_r2 = r2_score(y, y_pred)
        
        assert results["rmse"] == pytest.approx(expected_rmse)
        assert results["mse"] == pytest.approx(expected_mse)
        assert results["r2"] == pytest.approx(expected_r2)
    
    def test_cross_validate(self, classification_data):
        """Test cross-validation functionality."""
        X = classification_data["X"]
        y = classification_data["y"]
        model = classification_data["model"]
        
        # Cross-validate with accuracy
        evaluator = ModelEvaluator(metrics=["accuracy"])
        results = evaluator.cross_validate(model, X, y, cv=2)
        
        assert "accuracy" in results
        assert 0 <= results["accuracy"] <= 1
    
    def test_unsupported_metric(self, classification_data):
        """Test behavior with unsupported metrics."""
        X = classification_data["X"]
        y = classification_data["y"]
        model = classification_data["model"]
        
        # Try with an unsupported metric
        evaluator = ModelEvaluator(metrics=["unsupported_metric"])
        
        with pytest.raises(ValueError):
            evaluator.evaluate(model, X, y)