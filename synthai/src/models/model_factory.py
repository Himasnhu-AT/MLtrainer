"""
Model factory module for the SynthAI Model Training Framework.
This module creates and manages machine learning models.
"""
import os
import pickle
import time
from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import xgboost as xgb
from tqdm import tqdm
from synthai.src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

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
        logger.debug(f"Initialized {self.__class__.__name__} with params: {self.params}")
    
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
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        logger.debug(f"Making predictions on {X.shape[0]} samples")
        with tqdm(total=100, desc="Predicting", unit="%", ncols=100) as pbar:
            pbar.update(10)  # Starting prediction
            try:
                predictions = self.model.predict(X)
                pbar.update(90)  # Complete
                return predictions
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                pbar.set_description(f"Error: {str(e)}")
                raise
    
    @log_execution_time(logger)
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            logger.error("No model to save")
            raise ValueError("No model to save")
        
        logger.info(f"Saving model to {path}")
        with tqdm(total=1, desc="Saving model", unit="file") as pbar:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            pbar.update(1)
        logger.info(f"Model saved successfully to {path}")
    
    @classmethod
    @log_execution_time(logger)
    def load(cls, path: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from {path}")
        with tqdm(total=1, desc="Loading model", unit="file") as pbar:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            pbar.update(1)
        
        instance = cls()
        instance.model = model
        logger.info(f"Model loaded successfully from {path}")
        
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
        logger.info(f"Initializing {model_type} classification model")
        
        try:
            if model_type == "random_forest":
                self.model = RandomForestClassifier(**self.params)
            elif model_type == "logistic_regression":
                self.model = LogisticRegression(**self.params)
            elif model_type == "svm":
                self.model = SVC(**self.params)
            elif model_type == "xgboost":
                self.model = xgb.XGBClassifier(**self.params)
            else:
                logger.error(f"Unsupported classification model type: {model_type}")
                raise ValueError(f"Unsupported classification model type: {model_type}")
            
            logger.debug(f"Model initialized with parameters: {self.params}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    @log_execution_time(logger)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classification model."""
        logger.info(f"Training {self.model_type} classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        with tqdm(total=100, desc=f"Training {self.model_type}", unit="%", ncols=100) as pbar:
            # Training steps visualization
            start_time = time.time()
            pbar.update(5)  # Starting
            
            try:
                if self.model_type == "xgboost":
                    # For XGBoost, simply fit the model without callbacks
                    # as some versions might not support the callbacks parameter
                    pbar.set_description(f"Training {self.model_type}")
                    self.model.fit(X, y)
                    pbar.update(95)  # Complete training
                else:
                    # For other models, just show a simple progress bar
                    self.model.fit(X, y)
                    pbar.update(95)  # Complete main training
                
                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                # Log model info if available
                if hasattr(self.model, 'n_features_in_'):
                    logger.debug(f"Model trained on {self.model.n_features_in_} features")
                
                if hasattr(self.model, 'classes_'):
                    logger.debug(f"Model classes: {self.model.classes_}")
                
                if hasattr(self.model, 'feature_importances_'):
                    top5_indices = np.argsort(self.model.feature_importances_)[-5:]
                    logger.debug(f"Top 5 feature importance indices: {top5_indices}")
                
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                pbar.set_description(f"Error: {str(e)}")
                raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
        
        logger.debug(f"Getting probability estimates for {X.shape[0]} samples")
        with tqdm(total=100, desc="Predicting probabilities", unit="%", ncols=100) as pbar:
            pbar.update(10)  # Starting
            if hasattr(self.model, 'predict_proba'):
                try:
                    probs = self.model.predict_proba(X)
                    pbar.update(90)  # Complete
                    return probs
                except Exception as e:
                    logger.error(f"Error during probability prediction: {str(e)}")
                    pbar.set_description(f"Error: {str(e)}")
                    raise
            else:
                # For models that don't support predict_proba directly
                logger.error(f"predict_proba not supported for {self.model_type}")
                pbar.set_description("Not supported")
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
        logger.info(f"Initializing {model_type} regression model")
        
        try:
            if model_type == "random_forest":
                self.model = RandomForestRegressor(**self.params)
            elif model_type == "linear_regression":
                self.model = LinearRegression(**self.params)
            elif model_type == "svm":
                self.model = SVR(**self.params)
            elif model_type == "xgboost":
                self.model = xgb.XGBRegressor(**self.params)
            else:
                logger.error(f"Unsupported regression model type: {model_type}")
                raise ValueError(f"Unsupported regression model type: {model_type}")
            
            logger.debug(f"Model initialized with parameters: {self.params}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    @log_execution_time(logger)
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the regression model."""
        logger.info(f"Training {self.model_type} regressor on {X.shape[0]} samples with {X.shape[1]} features")
        
        with tqdm(total=100, desc=f"Training {self.model_type}", unit="%", ncols=100) as pbar:
            # Training steps visualization
            start_time = time.time()
            pbar.update(5)  # Starting
            
            try:
                if self.model_type == "xgboost":
                    # For XGBoost, simply fit the model without callbacks
                    # as some versions might not support the callbacks parameter
                    pbar.set_description(f"Training {self.model_type}")
                    self.model.fit(X, y)
                    pbar.update(95)  # Complete training
                else:
                    # For other models, just show a simple progress bar
                    self.model.fit(X, y)
                    pbar.update(95)  # Complete main training
                
                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                # Log model info if available
                if hasattr(self.model, 'n_features_in_'):
                    logger.debug(f"Model trained on {self.model.n_features_in_} features")
                
                if hasattr(self.model, 'feature_importances_'):
                    top5_indices = np.argsort(self.model.feature_importances_)[-5:]
                    logger.debug(f"Top 5 feature importance indices: {top5_indices}")
                
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                pbar.set_description(f"Error: {str(e)}")
                raise


class ModelFactory:
    """Factory class for creating models based on task and type."""
    
    def __init__(self):
        """Initialize the model factory."""
        logger.debug("ModelFactory initialized")
    
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
        logger.info(f"Creating model of type {model_type}")
        
        # Infer task from model_type if not provided
        if task is None:
            task = self._infer_task_from_model_type(model_type)
            logger.debug(f"Inferred task: {task}")
        else:
            logger.debug(f"Using provided task: {task}")
        
        # Create appropriate model
        try:
            if task == "classification":
                return ClassificationModel(params, model_type)
            elif task == "regression":
                return RegressionModel(params, model_type)
            else:
                logger.error(f"Unsupported task: {task}")
                raise ValueError(f"Unsupported task: {task}")
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
    
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
        available_models = {
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
        
        logger.debug(f"Available models: {available_models}")
        return available_models