"""
Data loader module for the SynthAI Model Training Framework.
This module handles loading data from various sources.
"""
import os
from typing import Optional

import pandas as pd


class DataLoader:
    """Class for loading data from various sources."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data file (currently supports CSV)
        """
        self.data_path = data_path
        
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path.
        
        Args:
            **kwargs: Additional arguments to pass to the appropriate loader function
            
        Returns:
            Pandas DataFrame containing the loaded data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        if file_ext == '.csv':
            return self._load_csv(**kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return self._load_excel(**kwargs)
        elif file_ext == '.json':
            return self._load_json(**kwargs)
        elif file_ext in ['.parquet', '.pq']:
            return self._load_parquet(**kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_csv(self, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file."""
        # Set some sensible defaults
        default_options = {
            'sep': ',',
            'header': 0,
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'NaN', 'None'],
            'low_memory': False
        }
        
        # Update with user-provided options
        for k, v in default_options.items():
            if k not in kwargs:
                kwargs[k] = v
        
        return pd.read_csv(self.data_path, **kwargs)
    
    def _load_excel(self, **kwargs) -> pd.DataFrame:
        """Load data from an Excel file."""
        return pd.read_excel(self.data_path, **kwargs)
    
    def _load_json(self, **kwargs) -> pd.DataFrame:
        """Load data from a JSON file."""
        return pd.read_json(self.data_path, **kwargs)
    
    def _load_parquet(self, **kwargs) -> pd.DataFrame:
        """Load data from a Parquet file."""
        return pd.read_parquet(self.data_path, **kwargs)
    
    @staticmethod
    def sample_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Return a sample of n rows from the dataframe.
        
        Args:
            df: Pandas DataFrame to sample from
            n: Number of rows to sample
            
        Returns:
            DataFrame with sampled rows
        """
        if n >= len(df):
            return df
        
        return df.sample(n=n, random_state=42)
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        """
        Generate a summary of the data.
        
        Args:
            df: Pandas DataFrame to summarize
            
        Returns:
            Dictionary containing summary information
        """
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "missing_percentage": {col: float(df[col].isnull().mean() * 100) for col in df.columns},
            "unique_values": {col: int(df[col].nunique()) for col in df.columns}
        }
        
        return summary