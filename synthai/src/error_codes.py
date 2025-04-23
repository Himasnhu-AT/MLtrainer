"""Error codes for the SynthAI package."""

# Define error codes as constants
SUCCESS = 0
ERROR_INVALID_ARGUMENT = 1
ERROR_MISSING_ENVIRONMENT = 2
ERROR_COMPATIBILITY = 3
ERROR_MISSING_FILE = 4
ERROR_TRAINING_FAILED = 5
ERROR_VALIDATION_FAILED = 6
ERROR_DATA_LOADING_FAILED = 7
ERROR_PREPROCESSING_FAILED = 8
ERROR_MODEL_CREATION_FAILED = 9
ERROR_EVALUATION_FAILED = 10
ERROR_SAVING_FAILED = 11
ERROR_MULTIPLE_FAILURES = 20

# Define error messages
ERROR_MESSAGES = {
    SUCCESS: "Success",
    ERROR_INVALID_ARGUMENT: "Invalid argument or option",
    ERROR_MISSING_ENVIRONMENT: "Required environment not found",
    ERROR_COMPATIBILITY: "Model and dataset are not compatible",
    ERROR_MISSING_FILE: "Required file not found",
    ERROR_TRAINING_FAILED: "Model training process failed",
    ERROR_VALIDATION_FAILED: "Schema validation failed",
    ERROR_DATA_LOADING_FAILED: "Failed to load data",
    ERROR_PREPROCESSING_FAILED: "Failed to preprocess data",
    ERROR_MODEL_CREATION_FAILED: "Failed to create model",
    ERROR_EVALUATION_FAILED: "Failed to evaluate model",
    ERROR_SAVING_FAILED: "Failed to save model",
    ERROR_MULTIPLE_FAILURES: "Multiple failures occurred"
}

def get_error_message(code):
    """Get the error message for a given error code.
    
    Args:
        code: The error code
        
    Returns:
        str: The error message
    """
    return ERROR_MESSAGES.get(code, f"Unknown error code: {code}")

class SynthAIError(Exception):
    """Base exception for SynthAI errors."""
    
    def __init__(self, message, error_code=ERROR_TRAINING_FAILED):
        """Initialize SynthAIError.
        
        Args:
            message: Error message
            error_code: Error code as defined in this module
        """
        self.error_code = error_code
        self.message = message
        super().__init__(f"Error {error_code}: {message}")

class ValidationError(SynthAIError):
    """Exception raised for schema validation errors."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_VALIDATION_FAILED)

class DataLoadingError(SynthAIError):
    """Exception raised for data loading errors."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_DATA_LOADING_FAILED)

class PreprocessingError(SynthAIError):
    """Exception raised for preprocessing errors."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_PREPROCESSING_FAILED)

class ModelCreationError(SynthAIError):
    """Exception raised for model creation errors."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_MODEL_CREATION_FAILED)

class CompatibilityError(SynthAIError):
    """Exception raised for model-data compatibility issues."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_COMPATIBILITY)

class EvaluationError(SynthAIError):
    """Exception raised for model evaluation errors."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_EVALUATION_FAILED)

class SavingError(SynthAIError):
    """Exception raised for model saving errors."""
    
    def __init__(self, message):
        super().__init__(message, ERROR_SAVING_FAILED)