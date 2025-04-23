"""
Model evaluator module for the SynthAI Model Training Framework.
This module evaluates model performance using various metrics.
"""
from typing import Dict, List, Any, Union

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the model evaluator.
        
        Args:
            metrics: List of metric names to calculate
        """
        self.metrics = metrics or ["accuracy"]
    
    def evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on the given data.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix
            y: True labels/values
            
        Returns:
            Dictionary of metric names to values
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Determine if this is a classification or regression task
        is_classification = self._is_classification_task(y)
        
        # Calculate each requested metric
        results = {}
        
        for metric in self.metrics:
            if is_classification:
                results[metric] = self._calculate_classification_metric(metric, y, y_pred)
            else:
                results[metric] = self._calculate_regression_metric(metric, y, y_pred)
        
        return results
    
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation and calculate metrics.
        
        Args:
            model: Untrained model instance
            X: Feature matrix
            y: True labels/values
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of metric names to values (averaged across folds)
        """
        results = {}
        
        # Determine if this is a classification or regression task
        is_classification = self._is_classification_task(y)
        
        for metric in self.metrics:
            # Convert metric name to sklearn scoring parameter
            scoring = self._convert_to_scoring_param(metric, is_classification)
            
            # Perform cross-validation
            scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring)
            
            # Store average score
            results[metric] = float(np.mean(scores))
        
        return results
    
    @staticmethod
    def _is_classification_task(y: np.ndarray) -> bool:
        """
        Determine if this is a classification task based on target values.
        
        Args:
            y: Target values
            
        Returns:
            True if classification, False if regression
        """
        return len(np.unique(y)) < 10 and np.array_equal(y, y.astype(int))
    
    def _calculate_classification_metric(self, metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate classification metric.
        
        Args:
            metric: Name of the metric
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Metric value
        """
        # Handle multi-class vs binary classification
        is_binary = len(np.unique(y_true)) == 2
        average = 'binary' if is_binary else 'weighted'
        
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif metric == "precision":
            return float(precision_score(y_true, y_pred, average=average, zero_division=0))
        elif metric == "recall":
            return float(recall_score(y_true, y_pred, average=average, zero_division=0))
        elif metric == "f1":
            return float(f1_score(y_true, y_pred, average=average, zero_division=0))
        else:
            raise ValueError(f"Unsupported classification metric: {metric}")
    
    def _calculate_regression_metric(self, metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate regression metric.
        
        Args:
            metric: Name of the metric
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "mse":
            return float(mean_squared_error(y_true, y_pred))
        elif metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        elif metric == "r2":
            return float(r2_score(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported regression metric: {metric}")
    
    @staticmethod
    def _convert_to_scoring_param(metric: str, is_classification: bool) -> str:
        """
        Convert our metric name to sklearn's scoring parameter name.
        
        Args:
            metric: Our metric name
            is_classification: Whether this is a classification task
            
        Returns:
            Sklearn scoring parameter name
        """
        if is_classification:
            # Classification metrics
            mapping = {
                "accuracy": "accuracy",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "f1": "f1_weighted"
            }
        else:
            # Regression metrics
            mapping = {
                "rmse": "neg_root_mean_squared_error",
                "mse": "neg_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2"
            }
        
        return mapping.get(metric, metric)