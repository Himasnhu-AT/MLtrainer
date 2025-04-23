"""
Data loader module for the SynthAI Model Training Framework.
This module handles loading data from various sources.
"""
import os
from typing import Optional
import pandas as pd
from tqdm import tqdm
import time
from synthai.src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

class DataLoader:
    """Class for loading data from various sources."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the data file (currently supports CSV)
        """
        self.data_path = data_path
        logger.info(f"DataLoader initialized with path: {data_path}")
        
    @log_execution_time(logger)
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path.
        
        Args:
            **kwargs: Additional arguments to pass to the appropriate loader function
            
        Returns:
            Pandas DataFrame containing the loaded data
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        file_ext = os.path.splitext(self.data_path)[1].lower()
        logger.info(f"Loading data from {self.data_path} (format: {file_ext})")
        
        # Show a progress bar while loading data
        with tqdm(total=100, desc="Loading data", unit="%", ncols=100) as pbar:
            pbar.update(10)  # Starting the data load process
            
            try:
                if file_ext == '.csv':
                    pbar.set_description("Reading CSV data")
                    df = self._load_csv(**kwargs)
                elif file_ext in ['.xlsx', '.xls']:
                    pbar.set_description("Reading Excel data")
                    df = self._load_excel(**kwargs)
                elif file_ext == '.json':
                    pbar.set_description("Reading JSON data")
                    df = self._load_json(**kwargs)
                elif file_ext in ['.parquet', '.pq']:
                    pbar.set_description("Reading Parquet data")
                    df = self._load_parquet(**kwargs)
                else:
                    logger.error(f"Unsupported file format: {file_ext}")
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                pbar.update(60)  # Data loaded successfully
                pbar.set_description("Analyzing data")
                
                # Log data summary
                shape = df.shape
                logger.info(f"Data loaded successfully: {shape[0]} rows, {shape[1]} columns")
                logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
                
                pbar.update(30)  # Complete
                pbar.set_description("Data loaded successfully")
                
                return df
                
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                pbar.set_description(f"Error: {str(e)}")
                raise
    
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
        
        logger.debug(f"Reading CSV with options: {kwargs}")
        return pd.read_csv(self.data_path, **kwargs)
    
    def _load_excel(self, **kwargs) -> pd.DataFrame:
        """Load data from an Excel file."""
        logger.debug(f"Reading Excel with options: {kwargs}")
        return pd.read_excel(self.data_path, **kwargs)
    
    def _load_json(self, **kwargs) -> pd.DataFrame:
        """Load data from a JSON file."""
        logger.debug(f"Reading JSON with options: {kwargs}")
        return pd.read_json(self.data_path, **kwargs)
    
    def _load_parquet(self, **kwargs) -> pd.DataFrame:
        """Load data from a Parquet file."""
        logger.debug(f"Reading Parquet with options: {kwargs}")
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
        logger.debug("Generating data summary")
        with tqdm(total=6, desc="Analyzing data", unit="metrics", ncols=100) as pbar:
            summary = {"shape": df.shape}
            pbar.update(1)
            
            summary["columns"] = list(df.columns)
            pbar.update(1)
            
            summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            pbar.update(1)
            
            pbar.set_description("Checking missing values")
            summary["missing_values"] = {col: int(df[col].isnull().sum()) for col in df.columns}
            pbar.update(1)
            
            summary["missing_percentage"] = {col: float(df[col].isnull().mean() * 100) for col in df.columns}
            pbar.update(1)
            
            pbar.set_description("Counting unique values")
            summary["unique_values"] = {col: int(df[col].nunique()) for col in df.columns}
            pbar.update(1)
        
        return summary