"""
Main pipeline module for the SynthAI Model Training Framework.
This module orchestrates the entire model training process.
"""
import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from synthai.src.schema.validator import SchemaValidator
from synthai.src.data.loader import DataLoader
from synthai.src.data.preprocessor import DataPreprocessor
from synthai.src.models.model_factory import ModelFactory
from synthai.src.models.evaluator import ModelEvaluator
from synthai.src.utils.logger import setup_logger
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
    
    return parser.parse_args()

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

def check_model_data_compatibility(model_type, schema):
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

def main():
    """Run the SynthAI model training pipeline."""
    args = parse_args()
    
    try:
        config = load_config(args.config)
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logger(f"synthai_{timestamp}", log_dir, level=args.log_level)
        
        logger.info("Starting SynthAI Model Training Pipeline")
        logger.info(f"Input data: {args.data}")
        logger.info(f"Schema file: {args.schema}")
        logger.info(f"Model type: {args.model_type}")
        
        # Load and validate schema
        logger.info("Validating schema...")
        try:
            schema_validator = SchemaValidator(args.schema)
            schema = schema_validator.load_schema()
        except FileNotFoundError:
            raise ValidationError(f"Schema file not found: {args.schema}")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in schema file: {e}")
        except Exception as e:
            raise ValidationError(f"Schema validation error: {str(e)}")
        
        # Check if model is compatible with dataset
        check_model_data_compatibility(args.model_type, schema)
        
        # Load data
        logger.info("Loading data...")
        try:
            data_loader = DataLoader(args.data)
            data = data_loader.load()
        except FileNotFoundError:
            raise DataLoadingError(f"Data file not found: {args.data}")
        except Exception as e:
            raise DataLoadingError(f"Error loading data: {str(e)}")
        
        # Validate data against schema
        try:
            schema_validator.validate_data(data)
        except Exception as e:
            raise ValidationError(f"Data does not match schema: {str(e)}")
        
        # Preprocess data
        logger.info("Preprocessing data...")
        try:
            preprocessor = DataPreprocessor(schema)
            X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
        except Exception as e:
            raise PreprocessingError(f"Error preprocessing data: {str(e)}")
        
        # Train model
        logger.info(f"Training {args.model_type} model...")
        try:
            model_factory = ModelFactory()
            model = model_factory.get_model(args.model_type, config.get("model_params", {}))
            model.fit(X_train, y_train)
        except ValueError as e:
            if "Unsupported" in str(e):
                raise ModelCreationError(f"Unsupported model type: {args.model_type}")
            elif "Invalid classes" in str(e):
                raise CompatibilityError(f"Model {args.model_type} is not compatible with the provided data: {e}")
            else:
                raise ModelCreationError(f"Error creating or training model: {str(e)}")
        except Exception as e:
            raise ModelCreationError(f"Error creating or training model: {str(e)}")
        
        # Evaluate model
        logger.info("Evaluating model...")
        try:
            evaluator = ModelEvaluator(metrics=config.get("metrics", ["accuracy"]))
            evaluation_results = evaluator.evaluate(model, X_test, y_test)
            
            for metric, value in evaluation_results.items():
                logger.info(f"{metric}: {value:.4f}")
        except Exception as e:
            raise EvaluationError(f"Error evaluating model: {str(e)}")
        
        # Save model
        try:
            model_dir = args.output
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{args.model_type}_{timestamp}.pkl")
            model.save(model_path)
            
            # Save preprocessor if requested
            if config.get("save_preprocessor", True):
                preprocessor_path = os.path.join(model_dir, f"preprocessor_{timestamp}.pkl")
                preprocessor.save(preprocessor_path)
            
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            raise SavingError(f"Error saving model or preprocessor: {str(e)}")
        
        logger.info("Pipeline completed successfully")
        
    except (ValidationError, DataLoadingError, PreprocessingError, 
            ModelCreationError, CompatibilityError, EvaluationError, SavingError) as e:
        # These will be caught by the CLI and proper error codes will be returned
        raise
    except Exception as e:
        # Unexpected exceptions
        raise

if __name__ == "__main__":
    main()