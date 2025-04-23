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
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                user_config = json.load(f)
            else:
                # Assuming YAML if not JSON
                import yaml
                user_config = yaml.safe_load(f)
        
        # Update default config with user-provided values
        default_config.update(user_config)
    
    return default_config


def main():
    """Run the SynthAI model training pipeline."""
    args = parse_args()
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
    
    try:
        # Load and validate schema
        logger.info("Validating schema...")
        schema_validator = SchemaValidator(args.schema)
        schema = schema_validator.load_schema()
        
        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(args.data)
        data = data_loader.load()
        
        # Validate data against schema
        schema_validator.validate_data(data)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(schema)
        X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
        
        # Train model
        logger.info(f"Training {args.model_type} model...")
        model_factory = ModelFactory()
        model = model_factory.get_model(args.model_type, config.get("model_params", {}))
        model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = ModelEvaluator(metrics=config.get("metrics", ["accuracy"]))
        evaluation_results = evaluator.evaluate(model, X_test, y_test)
        
        for metric, value in evaluation_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save model
        model_dir = args.output
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{args.model_type}_{timestamp}.pkl")
        model.save(model_path)
        
        # Save preprocessor if requested
        if config.get("save_preprocessor", True):
            preprocessor_path = os.path.join(model_dir, f"preprocessor_{timestamp}.pkl")
            preprocessor.save(preprocessor_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()