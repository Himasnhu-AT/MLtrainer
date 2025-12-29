"""
Model Tuner module for the SynthAI Model Training Framework.
This module handles hyperparameter tuning for models.
"""
from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from synthai.src.utils.logger import get_logger, log_execution_time
from synthai.src.models.model_factory import BaseModel

logger = get_logger(__name__)

class ModelTuner:
    """Handles hyperparameter tuning for models."""
    
    def __init__(self, model: BaseModel, param_grid: Dict[str, List[Any]], 
                 method: str = "grid", cv: int = 5, n_iter: int = 10, 
                 scoring: str = "accuracy", n_jobs: int = -1, verbose: int = 1):
        """
        Initialize the model tuner.
        
        Args:
            model: The base model to tune
            param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
            method: Tuning method ('grid' or 'random')
            cv: Number of folds for cross-validation
            n_iter: Number of parameter settings that are sampled (only for 'random' method)
            scoring: Strategy to evaluate the performance of the cross-validated model on the test set
            n_jobs: Number of jobs to run in parallel
            verbose: Controls the verbosity: the higher, the more messages
        """
        self.model = model
        self.param_grid = param_grid
        self.method = method
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        
        logger.info(f"Initialized ModelTuner with method={method}, cv={cv}, scoring={scoring}")
        
    @log_execution_time(logger)
    def tune(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing best parameters and score
        """
        logger.info(f"Starting hyperparameter tuning with {self.method} search...")
        
        # Get the underlying scikit-learn model
        if hasattr(self.model, 'model'):
            estimator = self.model.model
        else:
            estimator = self.model
            
        if self.method == "grid":
            search = GridSearchCV(
                estimator=estimator,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        elif self.method == "random":
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported tuning method: {self.method}")
            
        search.fit(X, y)
        
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.best_estimator = search.best_estimator_
        
        # Update the original model with the best estimator
        if hasattr(self.model, 'model'):
            self.model.model = self.best_estimator
            # Update model params with best params
            if hasattr(self.model, 'params'):
                self.model.params.update(self.best_params)
        
        logger.info(f"Tuning completed. Best score: {self.best_score:.4f}")
        logger.debug(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
