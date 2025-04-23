"""
SynthAI Model Training Framework
=================================

A robust and flexible framework for training machine learning models from CSV data
using a JSON schema definition.
"""

__version__ = "0.1.0"

# Import key components for easy access
from synthai.src.schema.validator import SchemaValidator
from synthai.src.data.loader import DataLoader
from synthai.src.data.preprocessor import DataPreprocessor
from synthai.src.models.model_factory import ModelFactory
from synthai.src.models.evaluator import ModelEvaluator
from synthai.src.utils.logger import setup_logger, get_logger

# Expose pipeline entrypoint
from synthai.src.pipeline import main as run_pipeline