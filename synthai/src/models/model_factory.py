"""
Model factory module for the SynthAI Model Training Framework.
This module creates and manages machine learning models.
"""
import os
import pickle
from typing import Dict, Any, Optional, Union, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import xgboost as xgb


class BaseModel:
    """Base model class with common functionality."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the base model.
        
        Args:
            params: Dictionary of model parameters
        """
        self.params = params or {}
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        instance = cls()
        instance.model = model
        
        return instance


class ClassificationModel(BaseModel):
    """Classification model class."""
    
    def __init__(self, params: Dict[str, Any] = None, model_type: str = "random_forest"):
        """
        Initialize the classification model.
        
        Args:
            params: Dictionary of model parameters
            model_type: Type of classification model to use
        """
        super().__init__(params)
        self.model_type = model_type
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(**self.params)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(**self.params)
        elif model_type == "svm":
            self.model = SVC(**self.params)
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            raise ValueError(f"Unsupported classification model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classification model."""
        self.model.fit(X, y)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models that don't support predict_proba directly
            raise NotImplementedError(f"predict_proba not supported for {self.model_type}")


class RegressionModel(BaseModel):
    """Regression model class."""
    
    def __init__(self, params: Dict[str, Any] = None, model_type: str = "random_forest"):
        """
        Initialize the regression model.
        
        Args:
            params: Dictionary of model parameters
            model_type: Type of regression model to use
        """
        super().__init__(params)
        self.model_type = model_type
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(**self.params)
        elif model_type == "linear_regression":
            self.model = LinearRegression(**self.params)
        elif model_type == "svm":
            self.model = SVR(**self.params)
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor(**self.params)
        else:
            raise ValueError(f"Unsupported regression model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the regression model."""
        self.model.fit(X, y)


class ModelFactory:
    """Factory class for creating models based on task and type."""
    
    def get_model(self, model_type: str, params: Dict[str, Any] = None, task: str = None) -> BaseModel:
        """
        Create and return a model instance based on type and task.
        
        Args:
            model_type: Type of model to create (e.g., random_forest, xgboost)
            params: Dictionary of model parameters
            task: Type of task (classification or regression); if None, inferred from model_type
            
        Returns:
            Model instance
        """
        # Normalize model type name
        model_type = model_type.lower()
        
        # Infer task from model_type if not provided
        if task is None:
            task = self._infer_task_from_model_type(model_type)
        
        # Create appropriate model
        if task == "classification":
            return ClassificationModel(params, model_type)
        elif task == "regression":
            return RegressionModel(params, model_type)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    @staticmethod
    def _infer_task_from_model_type(model_type: str) -> str:
        """
        Infer the task type from the model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Task type (classification or regression)
        """
        # Default to classification for most model types
        if "classifier" in model_type:
            return "classification"
        elif "regressor" in model_type:
            return "regression"
        
        # Specific cases
        if model_type in ["linear_regression", "svr"]:
            return "regression"
        
        # Default
        return "classification"
    
    @staticmethod
    def list_available_models() -> Dict[str, List[str]]:
        """
        List available models grouped by task.
        
        Returns:
            Dictionary of task to list of model types
        """
        return {
            "classification": [
                "random_forest",
                "logistic_regression",
                "svm",
                "xgboost"
            ],
            "regression": [
                "random_forest",
                "linear_regression",
                "svm",
                "xgboost"
            ]
        }