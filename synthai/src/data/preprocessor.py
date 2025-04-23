"""
Data preprocessor module for the SynthAI Model Training Framework.
This module handles data preprocessing based on schema definitions.
"""
import os
import pickle
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from synthai.src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

class DataPreprocessor:
    """Preprocesses data according to the schema definition."""
    
    def __init__(self, schema: Dict[str, Any], random_state: int = 42, test_size: float = 0.2):
        """
        Initialize the data preprocessor.
        
        Args:
            schema: The schema definition dictionary
            random_state: Random seed for reproducibility
            test_size: Proportion of data to use for testing
        """
        self.schema = schema
        self.random_state = random_state
        self.test_size = test_size
        
        # Initialize transformers dict to store preprocessing objects
        self.transformers = {}
        
        # Flag to disable stratification in tests
        self._use_stratify_for_test = True
        
        logger.info(f"DataPreprocessor initialized with test_size={test_size}, random_state={random_state}")
        logger.debug(f"Schema target: {schema.get('target', {}).get('name', 'not defined')}")
    
    @log_execution_time(logger)
    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data according to the schema and split into train/test sets.
        
        Args:
            data: The input DataFrame to preprocess
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Starting preprocessing of dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Extract feature names and target name from schema
        feature_names = [feature["name"] for feature in self.schema["features"]]
        target_name = self.schema["target"]["name"]
        
        # Split data into features and target
        X = data[feature_names].copy()
        y = data[target_name].copy()
        
        # Apply preprocessing to each feature based on its type and preprocessing directive
        processed_features = []
        
        with tqdm(total=len(feature_names) + 3, desc="Preprocessing data", unit="steps") as pbar:
            # Process each feature with progress tracking
            pbar.set_description("Processing features")
            for feature in self.schema["features"]:
                feature_name = feature["name"]
                feature_type = feature["type"]
                preprocessing = feature.get("preprocessing", None)
                
                pbar.set_description(f"Processing {feature_name} ({feature_type})")
                logger.debug(f"Processing feature '{feature_name}' of type '{feature_type}'")
                
                # Process feature based on its type and preprocessing directive
                if feature_type == "numeric":
                    processed_column = self._preprocess_numeric(X[feature_name], feature_name, preprocessing)
                elif feature_type == "categorical":
                    processed_column = self._preprocess_categorical(X[feature_name], feature_name, preprocessing)
                elif feature_type == "text":
                    processed_column = self._preprocess_text(X[feature_name], feature_name, preprocessing)
                elif feature_type == "datetime":
                    processed_column = self._preprocess_datetime(X[feature_name], feature_name, preprocessing)
                elif feature_type == "boolean":
                    processed_column = self._preprocess_boolean(X[feature_name], feature_name)
                else:
                    # Default behavior for unknown types - pass through as is
                    logger.warning(f"Unknown feature type '{feature_type}' for '{feature_name}', using raw values")
                    processed_column = X[feature_name].values.reshape(-1, 1)
                
                processed_features.append(processed_column)
                pbar.update(1)
            
            # Concatenate all processed features
            pbar.set_description("Combining features")
            if len(processed_features) > 1:
                X_processed = np.hstack(processed_features)
            else:
                X_processed = processed_features[0]
            pbar.update(1)
            
            # Process target variable
            pbar.set_description(f"Processing target: {target_name}")
            y_processed = self._preprocess_target(y)
            pbar.update(1)
            
            # Split into train/test sets
            pbar.set_description("Splitting into train/test sets")
            # Modified train_test_split to not use stratification for the test cases
            # Use stratification only when we have enough samples per class and not disabled by tests
            target_type = self.schema["target"]["type"]
            use_stratify = False
            
            if target_type in ["binary", "multiclass"] and self._use_stratify_for_test:
                # Check if we have enough samples in each class for stratification
                classes, counts = np.unique(y_processed, return_counts=True)
                min_class_count = np.min(counts)
                
                # We need at least 2 samples per class for stratification
                if min_class_count >= 2 and len(classes) * 2 <= len(y_processed):
                    use_stratify = True
                    logger.debug(f"Using stratified split with {len(classes)} classes")
                else:
                    logger.debug(f"Not using stratified split: min class count={min_class_count}, classes={len(classes)}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y_processed if use_stratify else None
            )
            pbar.update(1)
        
        # Log result summary
        logger.info(f"Preprocessing complete: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        if target_type in ["binary", "multiclass"]:
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            logger.debug(f"Train set class distribution: {dict(zip(train_classes, train_counts))}")
            test_classes, test_counts = np.unique(y_test, return_counts=True)
            logger.debug(f"Test set class distribution: {dict(zip(test_classes, test_counts))}")
        
        return X_train, X_test, y_train, y_test
    
    def _preprocess_numeric(self, series: pd.Series, name: str, preprocessing: Optional[str]) -> np.ndarray:
        """
        Preprocess numeric features.
        
        Args:
            series: The feature series to preprocess
            name: The name of the feature
            preprocessing: The preprocessing method to apply
            
        Returns:
            Preprocessed feature as numpy array
        """
        # Handle missing values
        series = series.fillna(series.mean())
        
        # Apply preprocessing if specified
        if preprocessing == "scale":
            scaler = StandardScaler()
            self.transformers[f"{name}_scaler"] = scaler
            return scaler.fit_transform(series.values.reshape(-1, 1))
        
        elif preprocessing == "minmax":
            scaler = MinMaxScaler()
            self.transformers[f"{name}_scaler"] = scaler
            return scaler.fit_transform(series.values.reshape(-1, 1))
        
        else:
            # Default: return as is
            return series.values.reshape(-1, 1)
    
    def _preprocess_categorical(self, series: pd.Series, name: str, preprocessing: Optional[str]) -> np.ndarray:
        """
        Preprocess categorical features.
        
        Args:
            series: The feature series to preprocess
            name: The name of the feature
            preprocessing: The preprocessing method to apply
            
        Returns:
            Preprocessed feature as numpy array
        """
        # Handle missing values
        series = series.fillna(series.mode()[0])
        
        # Apply preprocessing if specified
        if preprocessing == "one-hot":
            # Updated to use sparse_output instead of sparse
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.transformers[f"{name}_encoder"] = encoder
            return encoder.fit_transform(series.values.reshape(-1, 1))
        
        elif preprocessing == "label":
            encoder = LabelEncoder()
            self.transformers[f"{name}_encoder"] = encoder
            return encoder.fit_transform(series.values).reshape(-1, 1)
        
        else:
            # Default: label encoding
            encoder = LabelEncoder()
            self.transformers[f"{name}_encoder"] = encoder
            return encoder.fit_transform(series.values).reshape(-1, 1)
    
    def _preprocess_text(self, series: pd.Series, name: str, preprocessing: Optional[str]) -> np.ndarray:
        """
        Preprocess text features.
        
        Args:
            series: The feature series to preprocess
            name: The name of the feature
            preprocessing: The preprocessing method to apply
            
        Returns:
            Preprocessed feature as numpy array
        """
        # Handle missing values
        series = series.fillna("")
        
        # Apply preprocessing if specified
        if preprocessing == "tfidf":
            vectorizer = TfidfVectorizer(max_features=100)
            self.transformers[f"{name}_vectorizer"] = vectorizer
            return vectorizer.fit_transform(series).toarray()
        
        elif preprocessing == "count":
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=100)
            self.transformers[f"{name}_vectorizer"] = vectorizer
            return vectorizer.fit_transform(series).toarray()
        
        else:
            # Default: simple tfidf
            vectorizer = TfidfVectorizer(max_features=100)
            self.transformers[f"{name}_vectorizer"] = vectorizer
            return vectorizer.fit_transform(series).toarray()
    
    def _preprocess_datetime(self, series: pd.Series, name: str, preprocessing: Optional[str]) -> np.ndarray:
        """
        Preprocess datetime features.
        
        Args:
            series: The feature series to preprocess
            name: The name of the feature
            preprocessing: The preprocessing method to apply
            
        Returns:
            Preprocessed feature as numpy array
        """
        # Convert to datetime
        series = pd.to_datetime(series)
        
        # Handle missing values
        series = series.fillna(series.mode()[0])
        
        # Extract features from datetime
        if preprocessing == "components":
            # Extract year, month, day, day of week, hour, etc.
            year = series.dt.year.values.reshape(-1, 1)
            month = series.dt.month.values.reshape(-1, 1)
            day = series.dt.day.values.reshape(-1, 1)
            dayofweek = series.dt.dayofweek.values.reshape(-1, 1)
            
            return np.hstack([year, month, day, dayofweek])
        
        elif preprocessing == "timestamp":
            # Convert to Unix timestamp
            timestamp = series.astype(int).values // 10**9
            return timestamp.reshape(-1, 1)
        
        else:
            # Default: timestamp
            timestamp = series.astype(int).values // 10**9
            return timestamp.reshape(-1, 1)
    
    def _preprocess_boolean(self, series: pd.Series, name: str) -> np.ndarray:
        """
        Preprocess boolean features.
        
        Args:
            series: The feature series to preprocess
            name: The name of the feature
            
        Returns:
            Preprocessed feature as numpy array
        """
        # Handle missing values
        series = series.fillna(series.mode()[0])
        
        # Convert to 0/1
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int).values.reshape(-1, 1)
        
        # Try to convert string booleans
        if series.dtype == 'object':
            mapping = {'True': 1, 'False': 0, '1': 1, '0': 0, 1: 1, 0: 0, True: 1, False: 0}
            return series.map(mapping).astype(int).values.reshape(-1, 1)
        
        return series.astype(int).values.reshape(-1, 1)
    
    def _preprocess_target(self, series: pd.Series) -> np.ndarray:
        """
        Preprocess the target variable.
        
        Args:
            series: The target series to preprocess
            
        Returns:
            Preprocessed target as numpy array
        """
        target_type = self.schema["target"]["type"]
        
        # Handle missing values - this should ideally be done before
        series = series.dropna()
        
        if target_type == "binary" or target_type == "multiclass":
            encoder = LabelEncoder()
            self.transformers["target_encoder"] = encoder
            return encoder.fit_transform(series.values)
        
        elif target_type == "continuous":
            # For regression, just return as is
            return series.values
        
        # Default case
        return series.values
    
    @log_execution_time(logger)
    def save(self, path: str) -> None:
        """
        Save the preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        logger.info(f"Saving preprocessor to {path}")
        
        with tqdm(total=1, desc="Saving preprocessor", unit="file") as pbar:
            with open(path, 'wb') as f:
                pickle.dump({
                    'schema': self.schema,
                    'transformers': self.transformers,
                    'random_state': self.random_state,
                    'test_size': self.test_size
                }, f)
            pbar.update(1)
        
        logger.info(f"Preprocessor saved successfully to {path}")
    
    @classmethod
    @log_execution_time(logger)
    def load(cls, path: str) -> 'DataPreprocessor':
        """
        Load a preprocessor from disk.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        logger.info(f"Loading preprocessor from {path}")
        
        with tqdm(total=1, desc="Loading preprocessor", unit="file") as pbar:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            pbar.update(1)
        
        preprocessor = cls(
            schema=data['schema'],
            random_state=data['random_state'],
            test_size=data['test_size']
        )
        preprocessor.transformers = data['transformers']
        
        logger.info(f"Preprocessor loaded successfully from {path}")
        logger.debug(f"Loaded transformers: {list(preprocessor.transformers.keys())}")
        
        return preprocessor