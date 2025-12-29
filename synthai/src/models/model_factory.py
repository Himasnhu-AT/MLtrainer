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
        # Initialize metadata dictionary to track training information
        self.metadata = {
            'training_time': None,
            'training_date': None,
            'epochs': None,
            'iterations': None,
            'n_samples_trained': None,
            'n_features': None,
            'model_type': None,
            'task_type': None,
            'performance_history': [],
            'hyperparameters': self.params.copy()
        }
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
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update model metadata with new information.
        
        Args:
            key: Metadata key to update
            value: New value for the metadata
        """
        self.metadata[key] = value
        logger.debug(f"Updated metadata: {key}={value}")
    
    def update_performance_history(self, metrics: Dict[str, float], epoch: int = None, iteration: int = None) -> None:
        """
        Add performance metrics to the model's performance history.
        
        Args:
            metrics: Dictionary of metric names to values
            epoch: Current epoch number (optional)
            iteration: Current iteration number (optional)
        """
        entry = {
            'metrics': metrics,
            'timestamp': time.time()
        }
        
        if epoch is not None:
            entry['epoch'] = epoch
        
        if iteration is not None:
            entry['iteration'] = iteration
            
        self.metadata['performance_history'].append(entry)
        logger.debug(f"Added performance metrics to history: {metrics}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the model's metadata.
        
        Returns:
            Dictionary of model metadata
        """
        return self.metadata
    
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
            # Save both the model and metadata
            model_data = {
                'model': self.model,
                'metadata': self.metadata
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
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
                data = pickle.load(f)
            pbar.update(1)
        
        instance = cls()
        
        # Handle both new and old format model files
        if isinstance(data, dict) and 'model' in data and 'metadata' in data:
            # New format with metadata
            instance.model = data['model']
            instance.metadata = data['metadata']
            logger.debug("Loaded model with metadata")
        else:
            # Old format, just the model
            instance.model = data
            logger.debug("Loaded model without metadata (old format)")
        
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
            # Filter parameters based on model type
            filtered_params = self._filter_params_for_model(model_type, self.params)
            
            if model_type == "random_forest":
                self.model = RandomForestClassifier(**filtered_params)
            elif model_type == "logistic_regression":
                self.model = LogisticRegression(**filtered_params)
            elif model_type == "svm":
                self.model = SVC(**filtered_params)
            elif model_type == "xgboost":
                self.model = xgb.XGBClassifier(**filtered_params)
            else:
                logger.error(f"Unsupported classification model type: {model_type}")
                raise ValueError(f"Unsupported classification model type: {model_type}")
            
            # Update metadata with model type and task information
            self.update_metadata('model_type', model_type)
            self.update_metadata('task_type', 'classification')
            
            logger.debug(f"Model initialized with parameters: {filtered_params}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _filter_params_for_model(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters based on model type to ensure compatibility.
        
        Args:
            model_type: Type of model being created
            params: Original parameters dictionary
            
        Returns:
            Filtered parameters dictionary
        """
        if params is None:
            return {}
            
        filtered_params = params.copy()
        
        # Define parameters that are only applicable to specific model types
        neural_network_params = [
            "epochs", "batch_size", "learning_rate", "early_stopping",
            "early_stopping_patience", "validation_split"
        ]
        
        # Remove neural network params from scikit-learn models
        if model_type in ["random_forest", "logistic_regression", "linear_regression", "svm"]:
            for param in neural_network_params:
                if param in filtered_params:
                    filtered_params.pop(param)
                    logger.debug(f"Removed parameter '{param}' not applicable to {model_type}")
        
        # Handle XGBoost specific parameter mapping
        if model_type == "xgboost":
            # Map epochs to n_estimators for XGBoost
            if "epochs" in filtered_params and "n_estimators" not in filtered_params:
                filtered_params["n_estimators"] = filtered_params.pop("epochs")
                logger.debug("Mapped 'epochs' to 'n_estimators' for XGBoost")
            
            # Remove parameters not applicable to XGBoost
            for param in ["batch_size", "early_stopping", "validation_split"]:
                if param in filtered_params:
                    filtered_params.pop(param)
                    logger.debug(f"Removed parameter '{param}' not applicable to XGBoost")
                
        return filtered_params
    
    @log_execution_time(logger)
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = None, callbacks: List = None) -> None:
        """
        Train the classification model.
        
        Args:
            X: Feature matrix
            y: Target vector
            epochs: Number of training epochs (for iterative models like XGBoost)
            callbacks: List of callbacks for training (model-specific)
        """
        logger.info(f"Training {self.model_type} classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Update metadata before training
        self.update_metadata('n_samples_trained', X.shape[0])
        self.update_metadata('n_features', X.shape[1])
        self.update_metadata('training_date', time.strftime("%Y-%m-%d %H:%M:%S"))
        
        if epochs is not None:
            self.update_metadata('epochs', epochs)
        
        with tqdm(total=100, desc=f"Training {self.model_type}", unit="%", ncols=100) as pbar:
            # Training steps visualization
            start_time = time.time()
            pbar.update(5)  # Starting
            
            try:
                if self.model_type == "xgboost":
                    # For XGBoost, we can track epochs
                    pbar.set_description(f"Training {self.model_type}")
                    
                    # If epochs were specified and not already in model params
                    if epochs is not None and 'n_estimators' not in self.params:
                        self.model.n_estimators = epochs
                        
                    # Setup XGBoost-specific parameters
                    fit_params = {}
                    if callbacks:
                        fit_params['callbacks'] = callbacks
                    
                    # Store the number of actual epochs/estimators in metadata
                    self.update_metadata('epochs', self.model.n_estimators)
                    
                    self.model.fit(X, y, **fit_params)
                    pbar.update(95)  # Complete training
                else:
                    # For other models, just show a simple progress bar
                    self.model.fit(X, y)
                    
                    # Some models have iterations or similar concepts
                    if hasattr(self.model, 'n_iter_'):
                        self.update_metadata('iterations', self.model.n_iter_)
                    elif hasattr(self.model, 'n_estimators'):
                        self.update_metadata('iterations', self.model.n_estimators)
                        
                    pbar.update(95)  # Complete main training
                
                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                # Update metadata after training
                self.update_metadata('training_time', training_time)
                
                # Log model info if available
                if hasattr(self.model, 'n_features_in_'):
                    logger.debug(f"Model trained on {self.model.n_features_in_} features")
                    self.update_metadata('n_features', self.model.n_features_in_)
                
                if hasattr(self.model, 'classes_'):
                    logger.debug(f"Model classes: {self.model.classes_}")
                    self.update_metadata('classes', self.model.classes_.tolist())
                
                if hasattr(self.model, 'feature_importances_'):
                    top5_indices = np.argsort(self.model.feature_importances_)[-5:]
                    logger.debug(f"Top 5 feature importance indices: {top5_indices}")
                    self.update_metadata('feature_importances', self.model.feature_importances_.tolist())
                
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                pbar.set_description(f"Error: {str(e)}")
                raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probability predictions
        """
        if self.model is None:
            logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet")
            
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("Underlying model does not support predict_proba")
            
        return self.model.predict_proba(X)


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
            # Filter parameters based on model type
            filtered_params = self._filter_params_for_model(model_type, self.params)
            
            if model_type == "random_forest":
                self.model = RandomForestRegressor(**filtered_params)
            elif model_type == "linear_regression":
                self.model = LinearRegression(**filtered_params)
            elif model_type == "svm":
                self.model = SVR(**filtered_params)
            elif model_type == "xgboost":
                self.model = xgb.XGBRegressor(**filtered_params)
            else:
                logger.error(f"Unsupported regression model type: {model_type}")
                raise ValueError(f"Unsupported regression model type: {model_type}")
            
            # Update metadata with model type and task information
            self.update_metadata('model_type', model_type)
            self.update_metadata('task_type', 'regression')
            
            logger.debug(f"Model initialized with parameters: {filtered_params}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _filter_params_for_model(self, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters based on model type to ensure compatibility.
        
        Args:
            model_type: Type of model being created
            params: Original parameters dictionary
            
        Returns:
            Filtered parameters dictionary
        """
        if params is None:
            return {}
            
        filtered_params = params.copy()
        
        # Define parameters that are only applicable to specific model types
        neural_network_params = [
            "epochs", "batch_size", "learning_rate", "early_stopping",
            "early_stopping_patience", "validation_split"
        ]
        
        # Remove neural network params from scikit-learn models
        if model_type in ["random_forest", "logistic_regression", "linear_regression", "svm"]:
            for param in neural_network_params:
                if param in filtered_params:
                    filtered_params.pop(param)
                    logger.debug(f"Removed parameter '{param}' not applicable to {model_type}")
        
        # Handle XGBoost specific parameter mapping
        if model_type == "xgboost":
            # Map epochs to n_estimators for XGBoost
            if "epochs" in filtered_params and "n_estimators" not in filtered_params:
                filtered_params["n_estimators"] = filtered_params.pop("epochs")
                logger.debug("Mapped 'epochs' to 'n_estimators' for XGBoost")
            
            # Remove parameters not applicable to XGBoost
            for param in ["batch_size", "early_stopping", "validation_split"]:
                if param in filtered_params:
                    filtered_params.pop(param)
                    logger.debug(f"Removed parameter '{param}' not applicable to XGBoost")
        
        return filtered_params
    
    @log_execution_time(logger)
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = None, callbacks: List = None) -> None:
        """
        Train the regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            epochs: Number of training epochs (for iterative models like XGBoost)
            callbacks: List of callbacks for training (model-specific)
        """
        logger.info(f"Training {self.model_type} regressor on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Update metadata before training
        self.update_metadata('n_samples_trained', X.shape[0])
        self.update_metadata('n_features', X.shape[1])
        self.update_metadata('training_date', time.strftime("%Y-%m-%d %H:%M:%S"))
        
        if epochs is not None:
            self.update_metadata('epochs', epochs)
        
        with tqdm(total=100, desc=f"Training {self.model_type}", unit="%", ncols=100) as pbar:
            # Training steps visualization
            start_time = time.time()
            pbar.update(5)  # Starting
            
            try:
                if self.model_type == "xgboost":
                    # For XGBoost, we can track epochs
                    pbar.set_description(f"Training {self.model_type}")
                    
                    # If epochs were specified and not already in model params
                    if epochs is not None and 'n_estimators' not in self.params:
                        self.model.n_estimators = epochs
                    
                    # Setup XGBoost-specific parameters
                    fit_params = {}
                    if callbacks:
                        fit_params['callbacks'] = callbacks
                    
                    # Store the number of actual epochs/estimators in metadata
                    self.update_metadata('epochs', self.model.n_estimators)
                    
                    self.model.fit(X, y, **fit_params)
                    pbar.update(95)  # Complete training
                else:
                    # For other models, just show a simple progress bar
                    self.model.fit(X, y)
                    
                    # Some models have iterations or similar concepts
                    if hasattr(self.model, 'n_iter_'):
                        self.update_metadata('iterations', self.model.n_iter_)
                    elif hasattr(self.model, 'n_estimators'):
                        self.update_metadata('iterations', self.model.n_estimators)
                    
                    pbar.update(95)  # Complete main training
                
                training_time = time.time() - start_time
                logger.info(f"Model training completed in {training_time:.2f} seconds")
                
                # Update metadata after training
                self.update_metadata('training_time', training_time)
                
                # Log model info if available
                if hasattr(self.model, 'n_features_in_'):
                    logger.debug(f"Model trained on {self.model.n_features_in_} features")
                    self.update_metadata('n_features', self.model.n_features_in_)
                
                if hasattr(self.model, 'feature_importances_'):
                    top5_indices = np.argsort(self.model.feature_importances_)[-5:]
                    logger.debug(f"Top 5 feature importance indices: {top5_indices}")
                    self.update_metadata('feature_importances', self.model.feature_importances_.tolist())
                
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