"""
Schema validation module for the SynthAI Model Training Framework.
This module validates JSON schemas and ensures data conforms to them.
"""
import json
import os
from typing import Dict, Any, List

import jsonschema
import pandas as pd
from jsonschema import validate


class SchemaValidator:
    """Validates data schema and ensures data conforms to the schema."""
    
    # Define the meta-schema for our schema format
    META_SCHEMA = {
        "type": "object",
        "required": ["features", "target"],
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["numeric", "categorical", "text", "datetime", "boolean"]},
                        "preprocessing": {"type": "string"},
                        "nullable": {"type": "boolean"},
                        "constraints": {"type": "object"}
                    }
                }
            },
            "target": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["binary", "multiclass", "continuous"]}
                }
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "version": {"type": "string"},
                    "author": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["classification", "regression"]}
                }
            },
            "model_tracking": {
                "type": "object",
                "properties": {
                    "enable_metadata": {"type": "boolean"},
                    "track_performance_history": {"type": "boolean"},
                    "track_training_metrics": {"type": "boolean"},
                    "required_metadata": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "experiment_tracking": {
                        "type": "object",
                        "properties": {
                            "enable": {"type": "boolean"},
                            "tracking_server": {"type": ["string", "null"]},
                            "experiment_name": {"type": "string"}
                        }
                    }
                }
            },
            "training_params": {
                "type": "object",
                "properties": {
                    "test_size": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "random_state": {"type": "integer", "minimum": 0},
                    "epochs": {"type": "integer", "minimum": 1},
                    "n_samples_trained": {"type": "integer", "minimum": 0},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "learning_rate": {"type": "number", "minimum": 0.0},
                    "early_stopping": {"type": "boolean"},
                    "early_stopping_patience": {"type": "integer", "minimum": 1},
                    "validation_split": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "cross_validation": {"type": "boolean"},
                    "cv_folds": {"type": "integer", "minimum": 2},
                    "stratify": {"type": "boolean"}
                }
            }
        }
    }
    
    def __init__(self, schema_path: str):
        """
        Initialize the schema validator.
        
        Args:
            schema_path: Path to the JSON schema file
        """
        self.schema_path = schema_path
        self.schema = None
    
    def load_schema(self) -> Dict[str, Any]:
        """
        Load and validate the schema from the file.
        
        Returns:
            The loaded schema as a dictionary
        """
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            schema = json.load(f)
        
        # Validate the schema against our meta-schema
        try:
            validate(instance=schema, schema=self.META_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid schema format: {str(e)}")
        
        self.schema = schema
        return schema
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate if the given data conforms to the schema.
        
        Args:
            data: Pandas DataFrame containing the data to validate
            
        Returns:
            True if data validates against the schema, raises exception otherwise
        """
        if self.schema is None:
            self.load_schema()
        
        # Check if all required columns exist
        feature_names = [feature["name"] for feature in self.schema["features"]]
        target_name = self.schema["target"]["name"]
        
        required_columns = feature_names + [target_name]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing columns in data: {', '.join(missing_columns)}")
        
        # Validate each feature according to its type
        for feature in self.schema["features"]:
            col_name = feature["name"]
            col_type = feature["type"]
            
            # Skip validation if column is allowed to be null and contains null values
            if feature.get("nullable", False) and data[col_name].isnull().any():
                continue
            
            if col_type == "numeric":
                if not pd.api.types.is_numeric_dtype(data[col_name]):
                    # Try to convert if possible
                    try:
                        data[col_name] = pd.to_numeric(data[col_name])
                    except:
                        raise TypeError(f"Column '{col_name}' should be numeric")
            
            elif col_type == "categorical":
                # For categorical, we just ensure it's a string or numeric type
                # Actual encoding happens in preprocessing
                pass
            
            elif col_type == "datetime":
                # Try to convert to datetime
                try:
                    pd.to_datetime(data[col_name])
                except:
                    raise TypeError(f"Column '{col_name}' should be convertible to datetime")
            
            elif col_type == "boolean":
                # Check if boolean or convertible to boolean
                if not (data[col_name].isin([True, False, 0, 1, '0', '1', 'True', 'False']).all()):
                    raise TypeError(f"Column '{col_name}' should contain boolean values")
        
        # Validate target according to its type
        target_type = self.schema["target"]["type"]
        if target_type == "binary":
            unique_values = data[target_name].nunique()
            if unique_values > 2:
                raise ValueError(f"Target column '{target_name}' has {unique_values} unique values, expected 2 for binary classification")
        
        elif target_type == "multiclass":
            # Just ensure it's categorical, no special validation needed
            pass
        
        elif target_type == "continuous":
            if not pd.api.types.is_numeric_dtype(data[target_name]):
                # Try to convert if possible
                try:
                    data[target_name] = pd.to_numeric(data[target_name])
                except:
                    raise TypeError(f"Target column '{target_name}' should be numeric for regression")
        
        return True
    
    def get_training_params(self) -> Dict[str, Any]:
        """
        Get the training parameters from the schema.
        
        Returns:
            A dictionary containing training parameters or default values if not specified
        """
        if self.schema is None:
            self.load_schema()
            
        # Default training parameters
        default_params = {
            "test_size": 0.2,
            "random_state": 42,
            "epochs": 100,
            "batch_size": 32,
            "early_stopping": False,
            "validation_split": 0.1,
            "cross_validation": False,
            "stratify": True
        }
        
        # Update with schema-defined parameters if available
        if "training_params" in self.schema:
            default_params.update(self.schema["training_params"])
            
        return default_params
    
    def get_model_tracking_config(self) -> Dict[str, Any]:
        """
        Get the model tracking configuration from the schema.
        
        Returns:
            A dictionary containing model tracking configuration or default values if not specified
        """
        if self.schema is None:
            self.load_schema()
            
        # Default model tracking configuration
        default_config = {
            "enable_metadata": True,
            "track_performance_history": True,
            "track_training_metrics": True,
            "required_metadata": ["training_time", "epochs", "n_samples_trained", "n_features"],
            "experiment_tracking": {
                "enable": False,
                "tracking_server": None,
                "experiment_name": "default"
            }
        }
        
        # Update with schema-defined configuration if available
        if "model_tracking" in self.schema:
            # For nested dictionaries, we need to update them separately
            if "experiment_tracking" in self.schema["model_tracking"]:
                default_config["experiment_tracking"].update(
                    self.schema["model_tracking"].get("experiment_tracking", {})
                )
            
            # Remove experiment_tracking from copy to avoid double updating
            model_tracking_copy = self.schema["model_tracking"].copy()
            model_tracking_copy.pop("experiment_tracking", None)
            default_config.update(model_tracking_copy)
            
        return default_config
    
    @staticmethod
    def generate_schema_template() -> Dict[str, Any]:
        """
        Generate a template schema as a starting point.
        
        Returns:
            A dictionary containing a template schema
        """
        return {
            "features": [
                {"name": "feature1", "type": "numeric", "preprocessing": "scale"},
                {"name": "feature2", "type": "categorical", "preprocessing": "one-hot"},
                {"name": "feature3", "type": "text", "preprocessing": "tfidf"}
            ],
            "target": {"name": "target_column", "type": "binary"},
            "metadata": {
                "description": "Example schema for model training",
                "version": "1.0",
                "author": "SynthAI",
                "task_type": "classification"
            },
            "model_tracking": {
                "enable_metadata": True,
                "track_performance_history": True,
                "track_training_metrics": True,
                "required_metadata": [
                    "training_time", 
                    "epochs", 
                    "n_samples_trained", 
                    "n_features"
                ],
                "experiment_tracking": {
                    "enable": True,
                    "tracking_server": None,
                    "experiment_name": "default_experiment"
                }
            },
            "training_params": {
                "test_size": 0.2,
                "random_state": 42,
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping": True,
                "early_stopping_patience": 5,
                "validation_split": 0.1,
                "cross_validation": False,
                "cv_folds": 5,
                "stratify": True
            }
        }