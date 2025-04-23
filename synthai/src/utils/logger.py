"""
Logger utility module for the SynthAI Model Training Framework.
This module provides a standardized logging setup.
"""
import os
import sys
import logging
import inspect
import time
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any


class DebugFormatter(logging.Formatter):
    """
    Custom formatter that adds extra debug information when log level is DEBUG.
    """
    def format(self, record):
        # Add timestamp with milliseconds for detailed tracing
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
        milliseconds = int(record.created % 1 * 1000)
        
        # For debug level, add file, line number, and function name
        if record.levelno == logging.DEBUG:
            # Original message
            message = record.getMessage()
            
            # Get stack frame information for caller
            if not hasattr(record, 'filename') or not hasattr(record, 'lineno'):
                frame = inspect.currentframe()
                # Go up stack frames to find the caller (skip logging machinery)
                for _ in range(7):  # Adjust this number if needed
                    if frame is None:
                        break
                    frame = frame.f_back
                
                if frame:
                    filename = os.path.basename(frame.f_code.co_filename)
                    lineno = frame.f_lineno
                    func_name = frame.f_code.co_name
                else:
                    filename = "unknown"
                    lineno = 0
                    func_name = "unknown"
            else:
                filename = record.filename
                lineno = record.lineno
                func_name = record.funcName
            
            # Format with detailed info
            return f"{timestamp}.{milliseconds:03d} - {record.name} - {record.levelname} - [{filename}:{lineno} in {func_name}] - {message}"
        
        # For other levels, use standard format
        return f"{timestamp}.{milliseconds:03d} - {record.name} - {record.levelname} - {record.getMessage()}"


def setup_logger(name: str, log_dir: str = "logs", level: str = "INFO", 
                 log_format: str = None, detailed_debug: bool = True) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Custom log format string
        detailed_debug: Whether to use detailed debug formatting
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger and set level
    logger = logging.getLogger(name)
    
    # Map string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    numeric_level = level_map.get(level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create handlers
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10MB file size
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set handler levels
    file_handler.setLevel(numeric_level)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    # Always use DebugFormatter for consistency and easier debugging
    formatter = DebugFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log when logger is initialized
    if numeric_level == logging.DEBUG:
        logger.debug(f"Logger {name} initialized with level {level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger is not configured, set up with default settings
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


def log_method_call(logger, level=logging.DEBUG):
    """
    Decorator to log method entry and exit with parameters and return value.
    
    Args:
        logger: Logger to use
        level: Logging level for the decorated method
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Skip logging if logger is None
            if logger is None:
                return func(*args, **kwargs)
                
            # Only log if the logger level is sufficient
            if logger.isEnabledFor(level):
                # Get method info
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                
                # Log method entry
                logger.log(level, f"ENTER: {func.__name__}({signature})")
                
                # Call the function
                try:
                    result = func(*args, **kwargs)
                    # Log method exit with result
                    logger.log(level, f"EXIT: {func.__name__} -> {result!r}")
                    return result
                except Exception as e:
                    # Log exception
                    logger.log(level, f"EXCEPTION in {func.__name__}: {str(e)}")
                    raise
            else:
                # Just call the function without logging
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_execution_time(logger, level=logging.DEBUG):
    """
    Decorator to log method execution time.
    
    Args:
        logger: Logger to use
        level: Logging level for the decorated method
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Skip logging if logger is None
            if logger is None:
                return func(*args, **kwargs)
                
            # Only log if the logger level is sufficient
            if logger.isEnabledFor(level):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = (end_time - start_time) * 1000  # Convert to milliseconds
                logger.log(level, f"TIME: {func.__name__} took {duration:.2f}ms to execute")
                return result
            else:
                # Just call the function without logging
                return func(*args, **kwargs)
        return wrapper
    return decorator