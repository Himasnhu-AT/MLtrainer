"""
Main pipeline module for the SynthAI Model Training Framework.
This module orchestrates the entire model training process.
"""
import os
import json
import argparse
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from synthai.src.schema.validator import SchemaValidator
from synthai.src.data.loader import DataLoader
from synthai.src.data.preprocessor import DataPreprocessor
from synthai.src.models.model_factory import ModelFactory
from synthai.src.models.evaluator import ModelEvaluator
from synthai.src.models.tuner import ModelTuner
from synthai.src.utils.logger import setup_logger, log_method_call, log_execution_time
from synthai.src.error_codes import (
    ValidationError, 
    DataLoadingError,
    PreprocessingError,
    ModelCreationError,
    CompatibilityError,
    EvaluationError,
    SavingError
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SynthAI Model Training Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--schema", type=str, required=True, help="Path to JSON schema file")
    parser.add_argument("--model-type", type=str, default="random_forest", 
                        help="Type of model to train (e.g., random_forest, xgboost)")
    parser.add_argument("--output", type=str, default="models", 
                        help="Directory for saving the model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--iterations", type=int, default=1, 
                        help="Number of model training iterations to run")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    
    return parser.parse_args()

@log_execution_time(None)  # Logger will be assigned later
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file if provided, else use defaults."""
    default_config = {
        "test_size": 0.2,
        "random_state": 42,
        "model_params": {},
        "save_preprocessor": True,
        "validation_method": "cross_val",
        "cv_folds": 5,
        "metrics": ["accuracy", "f1", "precision", "recall"]
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    user_config = json.load(f)
                else:
                    # Assuming YAML if not JSON
                    import yaml
                    user_config = yaml.safe_load(f)
            
            # Update default config with user-provided values
            default_config.update(user_config)
        except json.JSONDecodeError:
            raise ValidationError(f"Invalid JSON format in config file: {config_path}")
        except yaml.YAMLError:
            raise ValidationError(f"Invalid YAML format in config file: {config_path}")
        except Exception as e:
            raise ValidationError(f"Error loading config file: {str(e)}")
    elif config_path:
        raise ValidationError(f"Config file not found: {config_path}")
    
    return default_config

@log_method_call(None)  # Logger will be assigned later
def check_model_data_compatibility(model_type: str, schema: Dict) -> None:
    """Check if the model type is compatible with the data schema."""
    task_type = schema.get("metadata", {}).get("task_type", "").lower()
    
    # Classification models
    classification_models = ["random_forest", "logistic_regression", "xgboost"]
    # Regression models
    regression_models = ["random_forest", "linear_regression", "xgboost"]
    
    if task_type == "classification":
        if model_type not in classification_models:
            raise CompatibilityError(
                f"Model type '{model_type}' is not compatible with classification tasks. "
                f"Use one of: {', '.join(classification_models)}"
            )
    elif task_type == "regression":
        if model_type not in regression_models:
            raise CompatibilityError(
                f"Model type '{model_type}' is not compatible with regression tasks. "
                f"Use one of: {', '.join(regression_models)}"
            )
        
        # Special case: logistic_regression can't be used for regression
        if model_type == "logistic_regression":
            raise CompatibilityError(
                "Logistic Regression cannot be used for regression tasks. "
                "Use Linear Regression instead."
            )
    else:
        raise ValidationError(
            f"Unknown task type: {task_type}. Expected 'classification' or 'regression'."
        )

def log_data_stats(logger: logging.Logger, data: pd.DataFrame, stage: str = "raw") -> None:
    """Log statistics about the data at various stages of the pipeline."""
    # Ensure logger is not None before checking isEnabledFor
    if logger is None or not logger.isEnabledFor(logging.DEBUG):
        return
    
    logger.debug(f"Data stats ({stage}):")
    logger.debug(f"  Shape: {data.shape}")
    logger.debug(f"  Columns: {list(data.columns)}")
    
    # Get basic statistics for each column
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            stats = data[col].describe()
            logger.debug(f"  Column '{col}' (numeric):")
            logger.debug(f"    min: {stats['min']:.2f}, max: {stats['max']:.2f}")
            logger.debug(f"    mean: {stats['mean']:.2f}, std: {stats['std']:.2f}")
            logger.debug(f"    null values: {data[col].isnull().sum()} ({data[col].isnull().mean()*100:.1f}%)")
        elif pd.api.types.is_categorical_dtype(data[col]) or data[col].nunique() < 20:
            logger.debug(f"  Column '{col}' (categorical):")
            logger.debug(f"    unique values: {data[col].nunique()}")
            logger.debug(f"    top 5 values: {data[col].value_counts().head(5).to_dict()}")
            logger.debug(f"    null values: {data[col].isnull().sum()} ({data[col].isnull().mean()*100:.1f}%)")
        else:
            logger.debug(f"  Column '{col}':")
            logger.debug(f"    type: {data[col].dtype}")
            logger.debug(f"    null values: {data[col].isnull().sum()} ({data[col].isnull().mean()*100:.1f}%)")

def log_processed_data_stats(logger: logging.Logger, 
                            X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Log statistics about the processed data splits."""
    # Ensure logger is not None before checking isEnabledFor
    if logger is None or not logger.isEnabledFor(logging.DEBUG):
        return
    
    logger.debug("Processed data stats:")
    logger.debug(f"  X_train shape: {X_train.shape}")
    logger.debug(f"  X_test shape: {X_test.shape}")
    logger.debug(f"  y_train shape: {y_train.shape}")
    logger.debug(f"  y_test shape: {y_test.shape}")
    
    # For classification tasks, log class distribution
    if len(np.unique(y_train)) < 20:  # Assume it's a classification task
        train_classes = np.unique(y_train, return_counts=True)
        test_classes = np.unique(y_test, return_counts=True)
        
        logger.debug("  Training class distribution:")
        for cls, count in zip(train_classes[0], train_classes[1]):
            pct = (count / len(y_train)) * 100
            logger.debug(f"    Class {cls}: {count} samples ({pct:.1f}%)")
        
        logger.debug("  Testing class distribution:")
        for cls, count in zip(test_classes[0], test_classes[1]):
            pct = (count / len(y_test)) * 100
            logger.debug(f"    Class {cls}: {count} samples ({pct:.1f}%)")

def log_model_details(logger: logging.Logger, model: Any, model_type: str) -> None:
    """Log details about the model configuration."""
    # Ensure logger is not None before checking isEnabledFor
    if logger is None or not logger.isEnabledFor(logging.DEBUG):
        return
    
    logger.debug(f"Model details ({model_type}):")
    
    # Get model parameters
    if hasattr(model, "get_params"):
        params = model.get_params()
        for key, value in params.items():
            logger.debug(f"  {key}: {value}")
    else:
        logger.debug("  No parameters available for this model")

def train_model_iteration(
    logger: logging.Logger,
    data: pd.DataFrame, 
    schema: Dict[str, Any], 
    model_type: str, 
    config: Dict[str, Any], 
    iteration: int,
    output_dir: str,
    tune: bool = False
) -> Dict[str, Any]:
    """Run a single iteration of model training and evaluation."""
    iteration_start_time = time.time()
    
    # Get training parameters from schema
    schema_validator = SchemaValidator(None)
    schema_validator.schema = schema
    schema_training_params = schema_validator.get_training_params()
    
    # Merge schema training params with config, giving priority to schema
    merged_config = config.copy()
    for key, value in schema_training_params.items():
        if key in merged_config:
            logger.debug(f"Using schema value for {key}: {value} (overriding config: {merged_config[key]})")
        merged_config[key] = value
    
    # Update random seed for each iteration to ensure diversity
    # Note: still using config random state as base to maintain compatibility
    iteration_config = merged_config.copy()
    iteration_config["random_state"] = config.get("random_state", 42) + iteration
    
    # Log iteration start
    logger.info(f"Starting iteration {iteration+1}")
    if logger is not None and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Iteration {iteration+1} config:")
        logger.debug(f"  Random seed: {iteration_config['random_state']}")
        logger.debug(f"  Testing using {iteration_config['test_size'] * 100:.1f}% of data")
        
        if "training_params" in schema:
            logger.debug(f"  Using training parameters from schema: {json.dumps(schema['training_params'], indent=2)}")
    
    try:
        # Preprocess data
        logger.info(f"Preprocessing data for iteration {iteration+1}...")
        preprocess_start = time.time()
        try:
            # Pass parameters from schema to preprocessor
            preprocessor = DataPreprocessor(
                schema, 
                random_state=iteration_config["random_state"], 
                test_size=iteration_config.get("test_size", 0.2)
            )
            X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
            
            # Get preprocessing metadata for model training
            preprocessing_metadata = preprocessor.get_preprocessing_metadata()
            
            if logger is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Data preprocessing completed in {(time.time() - preprocess_start) * 1000:.2f}ms")
                log_processed_data_stats(logger, X_train, X_test, y_train, y_test)
                logger.debug(f"Preprocessing metadata: {preprocessing_metadata}")
        except Exception as e:
            raise PreprocessingError(f"Error preprocessing data: {str(e)}")
        
        # Train model
        logger.info(f"Training {model_type} model (iteration {iteration+1})...")
        training_start = time.time()
        try:
            model_factory = ModelFactory()
            
            # Determine if this is a regression task
            task_type = schema.get("metadata", {}).get("task_type", "").lower()
            
            # Extract model-specific hyperparameters
            model_params = iteration_config.get("model_params", {})
            
            # Add training parameters from schema to model params
            for param in ["epochs", "batch_size", "learning_rate", "early_stopping", 
                          "early_stopping_patience", "validation_split"]:
                if param in iteration_config:
                    model_params[param] = iteration_config[param]
            
            model = model_factory.get_model(model_type, model_params, task=task_type)
            
            if logger is not None and logger.isEnabledFor(logging.DEBUG):
                log_model_details(logger, model, model_type)
                logger.debug(f"Starting model training with {X_train.shape[0]} samples...")
            
            if tune:
                # Perform hyperparameter tuning
                logger.info("Performing hyperparameter tuning...")
                
                # Define param grid based on model type (simplified example)
                # In a real scenario, this should come from config or schema
                param_grid = {}
                if model_type == "random_forest":
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10]
                    }
                elif model_type == "xgboost":
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7]
                    }
                # Add more models...
                
                if param_grid:
                    # Use random search if iterations > 1 (reusing iterations arg for n_iter in random search)
                    # This is a bit of a hack to reuse the arg
                    n_iter = iteration_config.get("iterations", 10)
                    method = "random" if n_iter > 1 else "grid"
                    
                    tuner = ModelTuner(model, param_grid, method=method, n_iter=n_iter, cv=3)
                    tune_results = tuner.tune(X_train, y_train)
                    
                    # Update model metadata with tuning results
                    if hasattr(model, 'update_metadata'):
                        model.update_metadata('tuned_params', tune_results['best_params'])
                        model.update_metadata('tuning_score', tune_results['best_score'])
                else:
                    logger.warning(f"No parameter grid defined for {model_type}, skipping tuning.")
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            # Update model metadata with preprocessing metadata
            if hasattr(model, 'update_metadata') and callable(model.update_metadata):
                for key, value in preprocessing_metadata.items():
                    model.update_metadata(key, value)
            
            if logger is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Model training completed in {(time.time() - training_start) * 1000:.2f}ms")
        except ValueError as e:
            if "Unsupported" in str(e):
                raise ModelCreationError(f"Unsupported model type: {model_type}")
            elif "Invalid classes" in str(e):
                raise CompatibilityError(f"Model {model_type} is not compatible with the provided data: {e}")
            else:
                raise ModelCreationError(f"Error creating or training model: {str(e)}")
        except Exception as e:
            raise ModelCreationError(f"Error creating or training model: {str(e)}")
        
        # Evaluate model
        logger.info(f"Evaluating model (iteration {iteration+1})...")
        evaluation_start = time.time()
        try:
            # Choose appropriate metrics based on task type
            task_type = schema.get("metadata", {}).get("task_type", "").lower()
            if task_type == "regression":
                metrics = ["rmse", "mae", "r2"]
            else:
                metrics = iteration_config.get("metrics", ["accuracy", "f1", "precision", "recall"])
            
            if logger is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Evaluating model with metrics: {metrics}")
            
            evaluator = ModelEvaluator(metrics=metrics)
            evaluation_results = evaluator.evaluate(model, X_test, y_test)
            
            if logger is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Model evaluation completed in {(time.time() - evaluation_start) * 1000:.2f}ms")
            
            # Modified part: Separate metadata from normal metrics when logging
            metadata = None
            if 'metadata' in evaluation_results:
                metadata = evaluation_results.pop('metadata')
            
            # Log each metric value with proper formatting (only for numeric values)
            for metric, value in evaluation_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"Iteration {iteration+1} - {metric}: {value:.4f}")
                else:
                    logger.info(f"Iteration {iteration+1} - {metric}: {value}")
            
            # Add metadata back to evaluation_results for return value if it was present
            if metadata is not None:
                evaluation_results['metadata'] = metadata
                
            # Log detailed model evaluation if in debug mode
            if logger is not None and logger.isEnabledFor(logging.DEBUG) and hasattr(evaluator, 'get_detailed_metrics'):
                detailed_metrics = evaluator.get_detailed_metrics()
                logger.debug(f"Detailed evaluation metrics (iteration {iteration+1}):")
                for metric_name, metric_value in detailed_metrics.items():
                    if isinstance(metric_value, dict):
                        logger.debug(f"  {metric_name}:")
                        for k, v in metric_value.items():
                            logger.debug(f"    {k}: {v}")
                    else:
                        logger.debug(f"  {metric_name}: {metric_value}")
                        
        except Exception as e:
            raise EvaluationError(f"Error evaluating model: {str(e)}")
        
        # Save model
        logger.info(f"Saving model (iteration {iteration+1})...")
        saving_start = time.time()
        try:
            model_dir = output_dir
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"{model_type}_iter{iteration+1}_{timestamp}.pkl")
            model.save(model_path)
            
            # Save preprocessor if requested
            if iteration_config.get("save_preprocessor", True):
                preprocessor_path = os.path.join(model_dir, f"preprocessor_iter{iteration+1}_{timestamp}.pkl")
                preprocessor.save(preprocessor_path)
            
            if logger is not None and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Model and preprocessor saved in {(time.time() - saving_start) * 1000:.2f}ms")
                logger.debug(f"Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
                if iteration_config.get("save_preprocessor", True):
                    logger.debug(f"Preprocessor file size: {os.path.getsize(preprocessor_path) / 1024:.2f} KB")
            
            logger.info(f"Model (iteration {iteration+1}) saved to {model_path}")
            
            iteration_time = time.time() - iteration_start_time
            logger.info(f"Iteration {iteration+1} completed in {iteration_time:.2f} seconds")
            
            # Return evaluation results for this iteration
            return {
                "iteration": iteration + 1,
                "model_path": model_path,
                "evaluation": evaluation_results,
                "training_time": iteration_time
            }
            
        except Exception as e:
            raise SavingError(f"Error saving model or preprocessor: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in iteration {iteration+1}: {str(e)}")
        raise

@log_execution_time(None)  # Logger will be assigned later
def main():
    """Run the SynthAI model training pipeline."""
    start_time = time.time()
    args = parse_args()
    
    # If verbose flag is set, use DEBUG level
    if args.verbose:
        args.log_level = "DEBUG"
    
    try:
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logger(f"synthai_{timestamp}", log_dir, level=args.log_level)
        
        logger.info("Starting SynthAI Model Training Pipeline")
        logger.info(f"Input data: {args.data}")
        logger.info(f"Schema file: {args.schema}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Number of iterations: {args.iterations}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pipeline configuration:")
            logger.debug(f"  Log level: {args.log_level}")
            logger.debug(f"  Output directory: {args.output}")
            logger.debug(f"  Config file: {args.config if args.config else 'None (using defaults)'}")
        
        # Load configuration
        config = load_config(args.config)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Loaded configuration:")
            for key, value in config.items():
                if isinstance(value, dict):
                    logger.debug(f"  {key}:")
                    for subkey, subvalue in value.items():
                        logger.debug(f"    {subkey}: {subvalue}")
                else:
                    logger.debug(f"  {key}: {value}")
        
        # Load and validate schema
        logger.info("Validating schema...")
        schema_start = time.time()
        try:
            schema_validator = SchemaValidator(args.schema)
            schema = schema_validator.load_schema()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Schema validation completed in {(time.time() - schema_start) * 1000:.2f}ms")
                logger.debug(f"Schema metadata:")
                for key, value in schema.get("metadata", {}).items():
                    logger.debug(f"  {key}: {value}")
                
                features = schema.get("features", [])
                logger.debug(f"Features defined in schema: {len(features)}")
                for i, feature in enumerate(features[:5]):  # Log first 5 features
                    logger.debug(f"  Feature {i+1}: {feature.get('name')} ({feature.get('type')})")
                
                if len(features) > 5:
                    logger.debug(f"  ... and {len(features) - 5} more features")
                
                target = schema.get("target", {})
                logger.debug(f"Target: {target.get('name')} ({target.get('type')})")
                
        except FileNotFoundError:
            raise ValidationError(f"Schema file not found: {args.schema}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in schema file: {e}")
        except Exception as e:
            raise ValidationError(f"Schema validation error: {str(e)}")
        
        # Check if model is compatible with dataset
        logger.debug("Checking model-data compatibility...")
        check_model_data_compatibility(args.model_type, schema)
        
        # Load data
        logger.info("Loading data...")
        data_loading_start = time.time()
        try:
            data_loader = DataLoader(args.data)
            data = data_loader.load()
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Data loading completed in {(time.time() - data_loading_start) * 1000:.2f}ms")
                log_data_stats(logger, data, "raw")
        except FileNotFoundError:
            raise DataLoadingError(f"Data file not found: {args.data}")
        except Exception as e:
            raise DataLoadingError(f"Error loading data: {str(e)}")
        
        # Validate data against schema
        logger.debug("Validating data against schema...")
        validation_start = time.time()
        try:
            schema_validator.validate_data(data)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Data validation completed in {(time.time() - validation_start) * 1000:.2f}ms")
        except Exception as e:
            raise ValidationError(f"Data does not match schema: {str(e)}")
        
        # Run multiple iterations if requested
        iteration_results = []
        for i in range(args.iterations):
            try:
                result = train_model_iteration(
                    logger=logger,
                    data=data,
                    schema=schema,
                    model_type=args.model_type,
                    config=config,
                    iteration=i,
                    output_dir=args.output,
                    tune=args.tune
                )
                iteration_results.append(result)
            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                if i == 0:  # If the first iteration fails, abort
                    raise
                else:
                    logger.warning(f"Continuing with next iteration despite error")
                    continue
        
        # Summarize results if multiple iterations
        if args.iterations > 1:
            logger.info("\n===== Iteration Summary =====")
            for result in iteration_results:
                iteration = result["iteration"]
                eval_metrics = result["evaluation"]
                time_taken = result["training_time"]
                
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
                logger.info(f"Iteration {iteration}: {metric_str} (time: {time_taken:.2f}s)")
            
            # Calculate average metrics across iterations
            if iteration_results:
                logger.info("\n===== Average Performance =====")
                metrics = {}
                for result in iteration_results:
                    for metric, value in result["evaluation"].items():
                        if metric not in metrics:
                            metrics[metric] = []
                        metrics[metric].append(value)
                
                for metric, values in metrics.items():
                    avg_value = sum(values) / len(values)
                    std_value = np.std(values) if len(values) > 1 else 0
                    logger.info(f"Average {metric}: {avg_value:.4f} ± {std_value:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        
    except (ValidationError, DataLoadingError, PreprocessingError, 
            ModelCreationError, CompatibilityError, EvaluationError, SavingError) as e:
        # These will be caught by the CLI and proper error codes will be returned
        raise
    except Exception as e:
        # Unexpected exceptions
        raise

if __name__ == "__main__":
    main()