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
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from synthai.src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

class DataPreprocessor:
    """Preprocesses data according to the schema definition."""
    
    def __init__(self, schema: Dict[str, Any], random_state: Optional[int] = None, test_size: Optional[float] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            schema: The schema definition dictionary
            random_state: Random seed for reproducibility (overrides schema if provided)
            test_size: Proportion of data to use for testing (overrides schema if provided)
        """
        self.schema = schema
        
        # Get training parameters from schema if available
        training_params = self._get_training_params_from_schema()
        
        # Use provided parameters if available, otherwise use schema values
        self.random_state = random_state if random_state is not None else training_params.get("random_state", 42)
        self.test_size = test_size if test_size is not None else training_params.get("test_size", 0.2)
        
        # Extract additional training parameters from schema
        self.stratify = training_params.get("stratify", True)
        self.cross_validation = training_params.get("cross_validation", False)
        self.cv_folds = training_params.get("cv_folds", 5)
        
        # Initialize transformers dict to store preprocessing objects
        self.transformers = {}
        
        # Flag to disable stratification in tests
        self._use_stratify_for_test = self.stratify
        
        logger.info(f"DataPreprocessor initialized with test_size={self.test_size}, random_state={self.random_state}")
        logger.debug(f"Schema target: {schema.get('target', {}).get('name', 'not defined')}")
        if "training_params" in schema:
            logger.debug(f"Using training parameters from schema: {training_params}")
    
    def _get_training_params_from_schema(self) -> Dict[str, Any]:
        """
        Extract training parameters from schema.
        
        Returns:
            Dictionary of training parameters
        """
        # Default training parameters
        default_params = {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "cross_validation": False,
            "cv_folds": 5
        }
        
        # Update with schema-defined parameters if available
        if "training_params" in self.schema:
            default_params.update(self.schema["training_params"])
        
        return default_params
    
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
            # Use stratification based on schema parameters and target type
            target_type = self.schema["target"]["type"]
            use_stratify = False
            
            if self._use_stratify_for_test and target_type in ["binary", "multiclass"]:
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
        
        # Store the number of samples for metadata tracking
        self.n_samples_trained = X_train.shape[0]
        self.n_features = X_train.shape[1]
        
        # Log result summary
        logger.info(f"Preprocessing complete: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        if target_type in ["binary", "multiclass"]:
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            logger.debug(f"Train set class distribution: {dict(zip(train_classes, train_counts))}")
            test_classes, test_counts = np.unique(y_test, return_counts=True)
            logger.debug(f"Test set class distribution: {dict(zip(test_classes, test_counts))}")
        
        return X_train, X_test, y_train, y_test
    
    def get_preprocessing_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the preprocessing.
        
        Returns:
            Dictionary containing preprocessing metadata
        """
        metadata = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify": self._use_stratify_for_test,
            "n_samples_trained": getattr(self, "n_samples_trained", 0),
            "n_features": getattr(self, "n_features", 0),
            "transformers": list(self.transformers.keys())
        }
        
        # Add additional training parameters if available
        if "training_params" in self.schema:
            for param, value in self.schema["training_params"].items():
                if param not in metadata:
                    metadata[param] = value
        
        return metadata
    
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
        # Handle missing values using SimpleImputer
        imputation = "mean" # Default
        if isinstance(preprocessing, dict) and "imputation" in preprocessing:
             imputation = preprocessing["imputation"]
        elif preprocessing == "impute_median":
             imputation = "median"
        elif preprocessing == "impute_mode":
             imputation = "most_frequent"
        
        # If preprocessing is just a string like "scale", we use default mean imputation
        # If it's a dict, we extract details
        
        imputer = SimpleImputer(strategy=imputation if imputation in ["mean", "median", "most_frequent", "constant"] else "mean")
        self.transformers[f"{name}_imputer"] = imputer
        series_imputed = imputer.fit_transform(series.values.reshape(-1, 1))
        
        # Apply scaling if specified
        if preprocessing == "scale" or (isinstance(preprocessing, dict) and preprocessing.get("scaling") == "standard"):
            scaler = StandardScaler()
            self.transformers[f"{name}_scaler"] = scaler
            return scaler.fit_transform(series_imputed)
        
        elif preprocessing == "minmax" or (isinstance(preprocessing, dict) and preprocessing.get("scaling") == "minmax"):
            scaler = MinMaxScaler()
            self.transformers[f"{name}_scaler"] = scaler
            return scaler.fit_transform(series_imputed)
        
        else:
            # Default: return imputed values
            return series_imputed
    
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
        imputation = "most_frequent" # Default
        if isinstance(preprocessing, dict) and "imputation" in preprocessing:
             imputation = preprocessing["imputation"]
        
        imputer = SimpleImputer(strategy=imputation if imputation in ["most_frequent", "constant"] else "most_frequent")
        self.transformers[f"{name}_imputer"] = imputer
        # For categorical, we need to ensure we're working with strings/objects for some imputers
        series_reshaped = series.values.reshape(-1, 1)
        series_imputed = imputer.fit_transform(series_reshaped)
        
        # Apply encoding if specified
        encoding = preprocessing if isinstance(preprocessing, str) else (preprocessing.get("encoding", "label") if isinstance(preprocessing, dict) else "label")
        
        if encoding == "one-hot":
            # Updated to use sparse_output instead of sparse
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.transformers[f"{name}_encoder"] = encoder
            return encoder.fit_transform(series_imputed)
        
        elif encoding == "label":
            encoder = LabelEncoder()
            self.transformers[f"{name}_encoder"] = encoder
            # LabelEncoder expects 1D array
            return encoder.fit_transform(series_imputed.ravel()).reshape(-1, 1)
        
        else:
            # Default: label encoding
            encoder = LabelEncoder()
            self.transformers[f"{name}_encoder"] = encoder
            return encoder.fit_transform(series_imputed.ravel()).reshape(-1, 1)
    
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
                    'test_size': self.test_size,
                    'stratify': self._use_stratify_for_test,
                    'cross_validation': self.cross_validation,
                    'cv_folds': self.cv_folds,
                    'n_samples_trained': getattr(self, "n_samples_trained", 0),
                    'n_features': getattr(self, "n_features", 0)
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
        preprocessor._use_stratify_for_test = data.get('stratify', True)
        preprocessor.cross_validation = data.get('cross_validation', False)
        preprocessor.cv_folds = data.get('cv_folds', 5)
        
        # Set sample and feature counts if available
        if 'n_samples_trained' in data:
            preprocessor.n_samples_trained = data['n_samples_trained']
        if 'n_features' in data:
            preprocessor.n_features = data['n_features']
        
        logger.info(f"Preprocessor loaded successfully from {path}")
        logger.debug(f"Loaded transformers: {list(preprocessor.transformers.keys())}")
        
        return preprocessor