"""
Unit tests for the data preprocessor module.
"""
import os
import tempfile
import pickle
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from synthai.src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_schema(self):
        """Fixture providing a sample schema."""
        return {
            "features": [
                {"name": "age", "type": "numeric", "preprocessing": "scale"},
                {"name": "income", "type": "numeric", "preprocessing": "minmax"},
                {"name": "category", "type": "categorical", "preprocessing": "one-hot"},
                {"name": "description", "type": "text", "preprocessing": "tfidf"}
            ],
            "target": {"name": "target", "type": "binary"}
        }
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data that matches the schema."""
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "category": ["A", "B", "A", "C", "B"],
            "description": [
                "This is a sample text",
                "Another example",
                "More text for testing",
                "Testing preprocessing",
                "Final sample"
            ],
            "target": [0, 1, 0, 1, 0]
        })
    
    def test_init(self, sample_schema):
        """Test the initialization of DataPreprocessor."""
        preprocessor = DataPreprocessor(sample_schema)
        
        assert preprocessor.schema == sample_schema
        assert preprocessor.random_state == 42  # Default value
        assert preprocessor.test_size == 0.2    # Default value
        assert isinstance(preprocessor.transformers, dict)
    
    def test_preprocess_shapes(self, sample_schema, sample_data):
        """Test that preprocessing returns arrays with expected shapes."""
        preprocessor = DataPreprocessor(sample_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Check that all arrays have proper dimensions
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        
        # Check train/test split ratios
        assert len(X_train) == 4  # 80% of data (rounded)
        assert len(X_test) == 1   # 20% of data
        assert len(y_train) == 4
        assert len(y_test) == 1
    
    def test_preprocess_numeric(self, sample_schema, sample_data):
        """Test preprocessing of numeric features."""
        # Modify schema to only include numeric features
        numeric_schema = {
            "features": [
                {"name": "age", "type": "numeric", "preprocessing": "scale"},
                {"name": "income", "type": "numeric", "preprocessing": "minmax"}
            ],
            "target": {"name": "target", "type": "binary"}
        }
        
        preprocessor = DataPreprocessor(numeric_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Check that the transformers were created
        assert "age_scaler" in preprocessor.transformers
        assert isinstance(preprocessor.transformers["age_scaler"], StandardScaler)
        
        # Check shape of output
        assert X_train.shape[1] == 2  # Two features
    
    def test_preprocess_categorical(self, sample_schema, sample_data):
        """Test preprocessing of categorical features."""
        # Modify schema to only include categorical feature
        cat_schema = {
            "features": [
                {"name": "category", "type": "categorical", "preprocessing": "one-hot"}
            ],
            "target": {"name": "target", "type": "binary"}
        }
        
        preprocessor = DataPreprocessor(cat_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Check that the transformer was created
        assert "category_encoder" in preprocessor.transformers
        assert isinstance(preprocessor.transformers["category_encoder"], OneHotEncoder)
        
        # Check shape of output - should be number of unique categories
        assert X_train.shape[1] == 3  # A, B, C categories
    
    def test_preprocess_text(self, sample_schema, sample_data):
        """Test preprocessing of text features."""
        # Modify schema to only include text feature
        text_schema = {
            "features": [
                {"name": "description", "type": "text", "preprocessing": "tfidf"}
            ],
            "target": {"name": "target", "type": "binary"}
        }
        
        preprocessor = DataPreprocessor(text_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Check that the transformer was created
        assert "description_vectorizer" in preprocessor.transformers
        
        # Shape should match the tfidf vectorizer's output
        assert X_train.shape[1] > 0  # Should have some features from text
    
    def test_save_load(self, sample_schema, sample_data):
        """Test saving and loading a preprocessor."""
        preprocessor = DataPreprocessor(sample_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Save preprocessor
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            preprocessor.save(f.name)
            
            # Load preprocessor
            loaded_preprocessor = DataPreprocessor.load(f.name)
            
            # Check attributes
            assert loaded_preprocessor.schema == preprocessor.schema
            assert loaded_preprocessor.random_state == preprocessor.random_state
            assert loaded_preprocessor.test_size == preprocessor.test_size
            assert set(loaded_preprocessor.transformers.keys()) == set(preprocessor.transformers.keys())
            
            os.unlink(f.name)
    
    def test_preprocess_binary_target(self, sample_schema, sample_data):
        """Test preprocessing of a binary target."""
        preprocessor = DataPreprocessor(sample_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Check that target was encoded correctly
        assert set(np.unique(y_train)).issubset({0, 1})
        assert "target_encoder" in preprocessor.transformers
    
    def test_preprocess_multiclass_target(self, sample_schema, sample_data):
        """Test preprocessing of a multiclass target."""
        # Modify schema to have multiclass target
        multiclass_schema = sample_schema.copy()
        multiclass_schema["target"] = {"name": "category", "type": "multiclass"}
        
        preprocessor = DataPreprocessor(multiclass_schema)
        preprocessor._use_stratify_for_test = False  # Disable stratification for testing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(sample_data)
        
        # Check that target was encoded correctly
        assert len(np.unique(y_train)) == len(sample_data["category"].unique())
        assert "target_encoder" in preprocessor.transformers