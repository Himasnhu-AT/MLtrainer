"""
Model Explainer module for the SynthAI Model Training Framework.
This module handles model explainability using SHAP.
"""
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from synthai.src.utils.logger import get_logger, log_execution_time
from synthai.src.models.model_factory import BaseModel

logger = get_logger(__name__)

class ModelExplainer:
    """Handles model explainability using SHAP."""
    
    def __init__(self, model: BaseModel, X_train: np.ndarray, feature_names: List[str] = None, task_type: str = "classification"):
        """
        Initialize the model explainer.
        
        Args:
            model: The trained model wrapper
            X_train: Training data used for initializing the explainer
            feature_names: List of feature names
            task_type: Type of task ('classification' or 'regression')
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.task_type = task_type
        self.explainer = None
        self.shap_values = None
        
        # Get the underlying scikit-learn model
        if hasattr(self.model, 'model'):
            self.estimator = self.model.model
        else:
            self.estimator = self.model
            
        logger.info("Initialized ModelExplainer")
        
    @log_execution_time(logger)
    def explain(self, X_test: np.ndarray, sample_size: int = 100) -> None:
        """
        Generate SHAP values for the test set.
        
        Args:
            X_test: Test data to explain
            sample_size: Number of samples to use for explanation (to speed up)
        """
        logger.info(f"Generating explanations for {min(len(X_test), sample_size)} samples...")
        
        # Subsample if necessary
        if len(X_test) > sample_size:
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_explain = X_test[indices]
        else:
            X_explain = X_test
            
        # Initialize appropriate explainer
        try:
            # TreeExplainer for tree-based models
            if hasattr(self.estimator, "feature_importances_"):
                self.explainer = shap.TreeExplainer(self.estimator)
                self.shap_values = self.explainer.shap_values(X_explain)
            # LinearExplainer for linear models
            elif hasattr(self.estimator, "coef_"):
                self.explainer = shap.LinearExplainer(self.estimator, self.X_train)
                self.shap_values = self.explainer.shap_values(X_explain)
            # KernelExplainer as fallback
            else:
                self.explainer = shap.KernelExplainer(self.estimator.predict, self.X_train[:50]) # Use small background
                self.shap_values = self.explainer.shap_values(X_explain)
                
            logger.info("SHAP values generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating SHAP values: {str(e)}")
            raise
            
    def save_plots(self, output_dir: str) -> List[str]:
        """
        Save explanation plots to the output directory.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of saved plot paths
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() before saving plots")
            
        os.makedirs(output_dir, exist_ok=True)
        saved_plots = []
        
        try:
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, features=self.X_train, feature_names=self.feature_names, show=False)
            summary_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(summary_path, bbox_inches='tight')
            plt.close()
            saved_plots.append(summary_path)
            
            # Bar plot (feature importance)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, features=self.X_train, feature_names=self.feature_names, plot_type="bar", show=False)
            bar_path = os.path.join(output_dir, "shap_importance.png")
            plt.savefig(bar_path, bbox_inches='tight')
            plt.close()
            saved_plots.append(bar_path)
            
            logger.info(f"Saved explanation plots to {output_dir}")
            return saved_plots
            
        except Exception as e:
            logger.error(f"Error saving plots: {str(e)}")
            # Don't raise, just return what we have
            return saved_plots
