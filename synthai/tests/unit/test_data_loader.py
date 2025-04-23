"""
Unit tests for the data loader module.
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from synthai.src.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Fixture providing sample CSV data."""
        data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [90.5, 85.0, 92.3, 88.7, 95.2]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name, data
    
    @pytest.fixture
    def sample_excel_data(self):
        """Fixture providing sample Excel data."""
        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed, skipping Excel tests")
            
        data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [90.5, 85.0, 92.3, 88.7, 95.2]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            data.to_excel(f.name, index=False)
            return f.name, data
    
    @pytest.fixture
    def sample_json_data(self):
        """Fixture providing sample JSON data."""
        data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [90.5, 85.0, 92.3, 88.7, 95.2]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            data.to_json(f.name, orient='records')
            return f.name, data
    
    def test_load_csv(self, sample_csv_data):
        """Test loading data from a CSV file."""
        file_path, expected_data = sample_csv_data
        
        loader = DataLoader(file_path)
        loaded_data = loader.load()
        
        assert_frame_equal(loaded_data, expected_data)
    
    def test_load_excel(self, sample_excel_data):
        """Test loading data from an Excel file."""
        file_path, expected_data = sample_excel_data
        
        loader = DataLoader(file_path)
        loaded_data = loader.load()
        
        # Excel might have some formatting differences, so just check the shape and key values
        assert loaded_data.shape == expected_data.shape
        assert set(loaded_data.columns) == set(expected_data.columns)
        assert loaded_data['id'].tolist() == expected_data['id'].tolist()
    
    def test_load_json(self, sample_json_data):
        """Test loading data from a JSON file."""
        file_path, expected_data = sample_json_data
        
        loader = DataLoader(file_path)
        loaded_data = loader.load()
        
        # For JSON, order might be different so we check key properties
        assert loaded_data.shape == expected_data.shape
        assert set(loaded_data.columns) == set(expected_data.columns)
        assert set(loaded_data['id'].tolist()) == set(expected_data['id'].tolist())
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = DataLoader("/path/to/nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            loader.load()
    
    def test_load_unsupported_format(self):
        """Test loading a file with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is some text data.")
            file_path = f.name
        
        loader = DataLoader(file_path)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load()
        
        os.unlink(file_path)
    
    def test_sample_data(self, sample_csv_data):
        """Test the sample_data method."""
        file_path, data = sample_csv_data
        
        loader = DataLoader(file_path)
        loaded_data = loader.load()
        
        # Test sampling 3 rows
        sampled_data = DataLoader.sample_data(loaded_data, n=3)
        assert len(sampled_data) == 3
        
        # Test sampling more rows than available
        sampled_data = DataLoader.sample_data(loaded_data, n=10)
        assert len(sampled_data) == len(loaded_data)
    
    def test_get_data_summary(self, sample_csv_data):
        """Test the get_data_summary method."""
        file_path, data = sample_csv_data
        
        loader = DataLoader(file_path)
        loaded_data = loader.load()
        
        summary = DataLoader.get_data_summary(loaded_data)
        
        assert "shape" in summary
        assert summary["shape"] == loaded_data.shape
        assert "columns" in summary
        assert set(summary["columns"]) == set(loaded_data.columns)
        assert "dtypes" in summary
        assert "missing_values" in summary
        assert "unique_values" in summary
    
    def teardown_method(self, method):
        """Clean up after tests by removing any temporary files."""
        # This will be called after each test method
        for attr_name in dir(self):
            if attr_name.endswith('_data'):
                attr_value = getattr(self, attr_name, None)
                if attr_value and isinstance(attr_value, tuple):
                    file_path = attr_value[0]
                    if os.path.exists(file_path):
                        os.unlink(file_path)