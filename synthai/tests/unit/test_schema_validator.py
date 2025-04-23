"""
Unit tests for the schema validator module.
"""
import os
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

from synthai.src.schema.validator import SchemaValidator


class TestSchemaValidator:
    """Test cases for SchemaValidator class."""
    
    @pytest.fixture
    def valid_schema(self):
        """Fixture providing a valid schema."""
        return {
            "features": [
                {"name": "age", "type": "numeric", "preprocessing": "scale"},
                {"name": "category", "type": "categorical", "preprocessing": "one-hot"}
            ],
            "target": {"name": "target", "type": "binary"}
        }
    
    @pytest.fixture
    def valid_schema_file(self, valid_schema):
        """Fixture providing a path to a valid schema file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(valid_schema, f)
            return f.name
    
    @pytest.fixture
    def valid_data(self):
        """Fixture providing valid data that matches the schema."""
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "category": ["A", "B", "A", "C", "B"],
            "target": [0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def invalid_schema(self):
        """Fixture providing an invalid schema."""
        return {
            "features": [
                {"name": "age", "preprocessing": "scale"}  # missing type
            ],
            "target": {"name": "target", "type": "binary"}
        }
    
    @pytest.fixture
    def invalid_schema_file(self, invalid_schema):
        """Fixture providing a path to an invalid schema file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json.dump(invalid_schema, f)
            return f.name
    
    def test_load_schema_valid(self, valid_schema_file, valid_schema):
        """Test loading a valid schema."""
        validator = SchemaValidator(valid_schema_file)
        loaded_schema = validator.load_schema()
        assert loaded_schema == valid_schema
    
    def test_load_schema_invalid(self, invalid_schema_file):
        """Test loading an invalid schema raises error."""
        validator = SchemaValidator(invalid_schema_file)
        with pytest.raises(ValueError):
            validator.load_schema()
    
    def test_load_schema_nonexistent_file(self):
        """Test loading a non-existent schema file raises error."""
        validator = SchemaValidator("/path/to/nonexistent.json")
        with pytest.raises(FileNotFoundError):
            validator.load_schema()
    
    def test_validate_data_valid(self, valid_schema_file, valid_data):
        """Test validating data that conforms to the schema."""
        validator = SchemaValidator(valid_schema_file)
        validator.load_schema()
        
        # This should not raise an exception
        assert validator.validate_data(valid_data) is True
    
    def test_validate_data_missing_columns(self, valid_schema_file):
        """Test validating data with missing columns."""
        validator = SchemaValidator(valid_schema_file)
        validator.load_schema()
        
        # Create data with missing columns
        invalid_data = pd.DataFrame({
            "age": [25, 30, 35],
            # missing 'category'
            "target": [0, 1, 0]
        })
        
        with pytest.raises(ValueError):
            validator.validate_data(invalid_data)
    
    def test_validate_data_wrong_types(self, valid_schema_file):
        """Test validating data with wrong column types."""
        validator = SchemaValidator(valid_schema_file)
        validator.load_schema()
        
        # Create data with wrong types
        invalid_data = pd.DataFrame({
            "age": ["twenty", "thirty", "forty"],  # strings instead of numbers
            "category": ["A", "B", "C"],
            "target": [0, 1, 0]
        })
        
        with pytest.raises(TypeError):
            validator.validate_data(invalid_data)
    
    def test_binary_target_too_many_classes(self, valid_schema_file):
        """Test validating binary target with too many classes."""
        validator = SchemaValidator(valid_schema_file)
        validator.load_schema()
        
        # Create data with multi-class target
        invalid_data = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "category": ["A", "B", "A", "C"],
            "target": [0, 1, 2, 3]  # more than 2 classes
        })
        
        with pytest.raises(ValueError):
            validator.validate_data(invalid_data)
    
    def test_generate_schema_template(self):
        """Test generating a schema template."""
        template = SchemaValidator.generate_schema_template()
        
        # Check basic structure
        assert "features" in template
        assert "target" in template
        assert "metadata" in template
        
        # Check that it has the required fields
        assert isinstance(template["features"], list)
        assert isinstance(template["target"], dict)
        assert "name" in template["target"]
        assert "type" in template["target"]
        
    def teardown_method(self, method):
        """Clean up after tests by removing any temporary files."""
        # This will be called after each test method
        for attr_name in dir(self):
            if attr_name.endswith('_file'):
                attr_value = getattr(self, attr_name, None)
                if attr_value and isinstance(attr_value, str) and os.path.exists(attr_value):
                    os.unlink(attr_value)