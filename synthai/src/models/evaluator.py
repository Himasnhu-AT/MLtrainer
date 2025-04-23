"""
Model evaluator module for the SynthAI Model Training Framework.
This module evaluates model performance using various metrics.
"""
from typing import Dict, List, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from synthai.src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the model evaluator.
        
        Args:
            metrics: List of metric names to calculate
        """
        self.metrics = metrics or ["accuracy"]
        logger.debug(f"ModelEvaluator initialized with metrics: {self.metrics}")
    
    @log_execution_time(logger)
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
        logger.info(f"Evaluating model on {X.shape[0]} samples")
        
        # Make predictions with progress bar
        with tqdm(total=100, desc="Evaluating model", unit="%", ncols=100) as pbar:
            pbar.update(10)  # Starting evaluation
            
            try:
                logger.debug("Making predictions for evaluation")
                y_pred = model.predict(X)
                pbar.update(30)  # Predictions complete
                
                # Determine if this is a classification or regression task
                # We'll first check based on the metrics requested
                is_regression_metrics = any(m in ["rmse", "mse", "mae", "r2"] for m in self.metrics)
                
                # If metrics don't clearly indicate, try to infer from target values
                if not is_regression_metrics:
                    is_classification = self._is_classification_task(y)
                else:
                    is_classification = False
                
                task_type = "classification" if is_classification else "regression"
                logger.debug(f"Detected task type: {task_type}")
                
                # Calculate each requested metric
                results = {}
                metrics_step = 60 / len(self.metrics)  # Remaining progress divided by number of metrics
                
                logger.debug(f"Calculating {len(self.metrics)} metrics")
                for metric in self.metrics:
                    pbar.set_description(f"Calculating {metric}")
                    
                    # Choose the right calculation method based on the metric type
                    if metric in ["rmse", "mse", "mae", "r2"]:
                        # These are regression metrics
                        results[metric] = self._calculate_regression_metric(metric, y, y_pred)
                    elif metric in ["accuracy", "precision", "recall", "f1"]:
                        # These are classification metrics
                        results[metric] = self._calculate_classification_metric(metric, y, y_pred)
                    else:
                        logger.error(f"Unsupported metric: {metric}")
                        raise ValueError(f"Unsupported metric: {metric}")
                    
                    pbar.update(metrics_step)
                    logger.debug(f"Metric {metric}: {results[metric]:.4f}")
                
                # Include model metadata in results if available, but keep it separate
                # from other metrics to avoid formatting issues
                if hasattr(model, 'get_metadata') and callable(model.get_metadata):
                    metadata = model.get_metadata()
                    # Store metadata separately to avoid including it in formatted output
                    results['metadata'] = metadata
                    
                    # Add evaluation sample size to metadata
                    if hasattr(model, 'update_metadata') and callable(model.update_metadata):
                        model.update_metadata('evaluation_sample_size', X.shape[0])
                    
                    # Update the model's performance history if possible
                    if hasattr(model, 'update_performance_history') and callable(model.update_performance_history):
                        # Create a copy of results without the metadata to avoid circular reference
                        metrics_only = {k: v for k, v in results.items() if k != 'metadata'}
                        model.update_performance_history(metrics_only)
                
                # Log only the numerical metrics, not the metadata
                metric_log_str = ', '.join([f'{k}={v:.4f}' for k, v in results.items() if k != 'metadata' and isinstance(v, (int, float))])
                logger.info(f"Evaluation complete. Results: {metric_log_str}")
                
                return results
                
            except Exception as e:
                logger.error(f"Error during model evaluation: {str(e)}")
                pbar.set_description(f"Error: {str(e)}")
                raise
    
    @log_execution_time(logger)
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
        logger.info(f"Performing {cv}-fold cross-validation on {X.shape[0]} samples")
        results = {}
        
        # Determine task type based on metrics
        is_regression_metrics = any(m in ["rmse", "mse", "mae", "r2"] for m in self.metrics)
        
        # If metrics don't clearly indicate, try to infer from target values
        if not is_regression_metrics:
            is_classification = self._is_classification_task(y)
        else:
            is_classification = False
            
        task_type = "classification" if is_classification else "regression"
        logger.debug(f"Cross-validation for {task_type} task")
        
        # Total steps: metrics * folds
        total_steps = len(self.metrics) * cv
        with tqdm(total=total_steps, desc="Cross-validation", unit="fold") as pbar:
            
            for metric in self.metrics:
                # Convert metric name to sklearn scoring parameter
                scoring = self._convert_to_scoring_param(metric, is_classification)
                logger.debug(f"Using scoring parameter '{scoring}' for metric '{metric}'")
                
                try:
                    # Perform cross-validation
                    pbar.set_description(f"CV for {metric}")
                    
                    # We'll create a custom splitter to update the progress bar
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                    scores = []
                    
                    for train_idx, test_idx in kf.split(X):
                        # Train and evaluate on this fold
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        # Clone the model for this fold
                        from sklearn.base import clone
                        fold_model = clone(model.model)
                        
                        # Train
                        fold_model.fit(X_train, y_train)
                        
                        # Score (using sklearn's scorer)
                        from sklearn.metrics import get_scorer
                        scorer = get_scorer(scoring)
                        score = scorer(fold_model, X_test, y_test)
                        scores.append(score)
                        
                        # Update progress bar for this fold
                        pbar.update(1)
                    
                    # Store average score
                    mean_score = float(np.mean(scores))
                    results[metric] = mean_score
                    logger.debug(f"Cross-validation {metric}: {mean_score:.4f} (std: {np.std(scores):.4f})")
                    
                except Exception as e:
                    logger.error(f"Error during cross-validation for metric {metric}: {str(e)}")
                    pbar.set_description(f"Error: {str(e)}")
                    # Continue with other metrics
                    continue
        
        # Include model metadata for cross-validation 
        if hasattr(model, 'get_metadata') and callable(model.get_metadata):
            metadata = model.get_metadata()
            # Add cross-validation specific information
            cv_metadata = {
                'cv_folds': cv,
                'cv_sample_size': X.shape[0],
                'cv_features': X.shape[1]
            }
            metadata.update(cv_metadata)
            results['metadata'] = metadata
        
        # Log only the numerical metrics, not the metadata
        metric_log_str = ', '.join([f'{k}={v:.4f}' for k, v in results.items() if k != 'metadata' and isinstance(v, (int, float))])
        logger.info(f"Cross-validation complete. Results: {metric_log_str}")
        
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
        is_classification = len(np.unique(y)) < 10 and np.array_equal(y, y.astype(int))
        logger.debug(f"Task classification detection: {'classification' if is_classification else 'regression'}")
        return is_classification
    
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
        logger.debug(f"Calculating {metric} for {'binary' if is_binary else 'multi-class'} classification")
        
        try:
            if metric == "accuracy":
                return float(accuracy_score(y_true, y_pred))
            elif metric == "precision":
                return float(precision_score(y_true, y_pred, average=average, zero_division=0))
            elif metric == "recall":
                return float(recall_score(y_true, y_pred, average=average, zero_division=0))
            elif metric == "f1":
                return float(f1_score(y_true, y_pred, average=average, zero_division=0))
            else:
                logger.error(f"Unsupported classification metric: {metric}")
                raise ValueError(f"Unsupported classification metric: {metric}")
        except Exception as e:
            logger.error(f"Error calculating {metric}: {str(e)}")
            raise
    
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
        logger.debug(f"Calculating {metric} for regression")
        
        try:
            if metric == "rmse":
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric == "mse":
                return float(mean_squared_error(y_true, y_pred))
            elif metric == "mae":
                return float(mean_absolute_error(y_true, y_pred))
            elif metric == "r2":
                return float(r2_score(y_true, y_pred))
            else:
                logger.error(f"Unsupported regression metric: {metric}")
                raise ValueError(f"Unsupported regression metric: {metric}")
        except Exception as e:
            logger.error(f"Error calculating {metric}: {str(e)}")
            raise
    
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
        
        result = mapping.get(metric, metric)
        logger.debug(f"Converted metric '{metric}' to scoring parameter '{result}'")
        return result
        
    def get_model_performance_comparison(self, models_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Compare performance across multiple models.
        
        Args:
            models_results: Dictionary mapping model names to their evaluation results
            
        Returns:
            Dictionary containing performance comparison metrics and metadata
        """
        logger.info(f"Comparing performance of {len(models_results)} models")
        
        comparison = {}
        common_metrics = set()
        models_metadata = {}
        
        # Find common metrics across all models
        for model_name, results in models_results.items():
            metrics_set = {k for k in results.keys() if k != 'metadata'}
            if not common_metrics:
                common_metrics = metrics_set
            else:
                common_metrics = common_metrics.intersection(metrics_set)
            
            # Extract metadata if available
            if 'metadata' in results:
                models_metadata[model_name] = results['metadata']
        
        logger.debug(f"Common metrics for comparison: {common_metrics}")
        
        # Compare each common metric
        for metric in common_metrics:
            metric_values = {model_name: results[metric] 
                             for model_name, results in models_results.items()}
            
            # Find best model for this metric
            if metric in ["rmse", "mse", "mae"]:  # Lower is better
                best_model = min(metric_values.items(), key=lambda x: x[1])[0]
            else:  # Higher is better
                best_model = max(metric_values.items(), key=lambda x: x[1])[0]
            
            comparison[metric] = {
                'values': metric_values,
                'best_model': best_model,
                'best_value': metric_values[best_model]
            }
        
        # Add metadata comparison if available
        if models_metadata:
            # Analyze training parameters
            training_params = {
                model_name: {
                    'training_time': metadata.get('training_time'),
                    'n_samples_trained': metadata.get('n_samples_trained'),
                    'epochs': metadata.get('epochs'),
                    'iterations': metadata.get('iterations')
                } for model_name, metadata in models_metadata.items()
            }
            
            comparison['metadata_comparison'] = {
                'training_parameters': training_params
            }
            
            # Identify potential improvements by analyzing metadata
            model_improvement_suggestions = {}
            for model_name, metadata in models_metadata.items():
                suggestions = []
                
                # Check for potential improvements based on training data and parameters
                if metadata.get('epochs') is not None and metadata.get('epochs') < 100:
                    suggestions.append("Consider increasing number of epochs for better convergence")
                
                if metadata.get('n_samples_trained') is not None and metadata.get('n_samples_trained') < 1000:
                    suggestions.append("Consider collecting more training data")
                
                # Add model-specific suggestions
                model_type = metadata.get('model_type')
                if model_type == 'random_forest' and metadata.get('hyperparameters', {}).get('n_estimators', 0) < 100:
                    suggestions.append("Consider increasing n_estimators for Random Forest")
                
                if model_type == 'xgboost':
                    if 'max_depth' not in metadata.get('hyperparameters', {}):
                        suggestions.append("Consider tuning max_depth parameter for XGBoost")
                    
                if suggestions:
                    model_improvement_suggestions[model_name] = suggestions
            
            if model_improvement_suggestions:
                comparison['improvement_suggestions'] = model_improvement_suggestions
        
        logger.info("Performance comparison complete")
        return comparison