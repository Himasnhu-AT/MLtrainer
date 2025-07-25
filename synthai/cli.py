"""
Command-line interface for the SynthAI Model Training Framework.
This module provides a simple CLI for using the framework.
"""
import os
import logging
import sys
import typer
from typing import Optional, List

from synthai.src.pipeline import main as run_pipeline
from synthai.src.utils.logger import setup_logger
from synthai.src.schema.validator import SchemaValidator
from synthai.src.models.model_factory import ModelFactory
from synthai.src.error_codes import (
    SUCCESS,
    ERROR_INVALID_ARGUMENT,
    ERROR_TRAINING_FAILED,
    ERROR_VALIDATION_FAILED,
    ERROR_MODEL_CREATION_FAILED,
    get_error_message,
    SynthAIError,
    ValidationError,
    DataLoadingError,
    PreprocessingError,
    ModelCreationError,
    CompatibilityError,
    EvaluationError,
    SavingError
)

app = typer.Typer(help="SynthAI: A ML model training framework")
logger = setup_logger("synthai_cli")

@app.command()
def train(
    data: str = typer.Argument(..., help="Path to the input CSV file"),
    schema: str = typer.Argument(..., help="Path to the JSON schema file"),
    model_type: str = typer.Option("random_forest", help="Type of model to train"),
    output: str = typer.Option("models", help="Directory for saving the model"),
    config: Optional[str] = typer.Option(None, help="Path to optional configuration file"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output (same as --log-level DEBUG)"),
    iterations: int = typer.Option(1, help="Number of training iterations to run")
):
    """
    Train a machine learning model using the specified data and schema.
    """
    try:
        # Use verbose flag to override log_level if specified
        if verbose:
            log_level = "DEBUG"
        
        # Set up logger with specified log level
        global logger
        logger = setup_logger("synthai_cli", level=log_level)
        
        # Log detailed info in debug mode
        if log_level.upper() == "DEBUG":
            logger.debug(f"Command line arguments:")
            logger.debug(f"  data: {data}")
            logger.debug(f"  schema: {schema}")
            logger.debug(f"  model_type: {model_type}")
            logger.debug(f"  output: {output}")
            logger.debug(f"  config: {config}")
            logger.debug(f"  log_level: {log_level}")
            logger.debug(f"  verbose: {verbose}")
            logger.debug(f"  iterations: {iterations}")
        
        # Set up arguments for the pipeline
        logger.info(f"Training model with data: {data}, schema: {schema}, model type: {model_type}")
        logger.info(f"Output directory: {output}, log level: {log_level}")
        
        # Instead of modifying sys.argv directly, we'll create a new array with the arguments
        # and use it when running the pipeline
        pipeline_args = [
            "pipeline.py",
            "--data", data,
            "--schema", schema,
            "--model-type", model_type,
            "--output", output,
            "--log-level", log_level,
            "--iterations", str(iterations)
        ]
        
        if config:
            pipeline_args.extend(["--config", config])
        
        if verbose:
            pipeline_args.append("--verbose")
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # Replace sys.argv with our actual arguments
            sys.argv = pipeline_args
            
            # Run the pipeline
            logger.info("Starting the training pipeline")
            run_pipeline()
            logger.info("Training completed successfully")
            return SUCCESS
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
        
    except ValidationError as e:
        logger.error(f"Schema validation error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except DataLoadingError as e:
        logger.error(f"Data loading error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except ModelCreationError as e:
        logger.error(f"Model creation error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except CompatibilityError as e:
        logger.error(f"Compatibility error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except EvaluationError as e:
        logger.error(f"Evaluation error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except SavingError as e:
        logger.error(f"Model saving error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except SynthAIError as e:
        logger.error(f"SynthAI error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        typer.echo(f"ERROR {ERROR_TRAINING_FAILED}: {str(e)}", err=True)
        raise typer.Exit(code=ERROR_TRAINING_FAILED)

@app.command()
def generate_schema(
    output: str = typer.Argument(..., help="Path to save the schema template"),
    log_level: str = typer.Option("INFO", help="Logging level")
):
    """
    Generate a sample schema template.
    """
    # Update logger with specified log level
    global logger
    logger = setup_logger("synthai_cli", level=log_level)
    
    try:
        # Generate template schema
        schema = SchemaValidator.generate_schema_template()
        
        # Save to file
        import json
        with open(output, 'w') as f:
            json.dump(schema, f, indent=2)
        
        logger.info(f"Schema template saved to {output}")
        return SUCCESS
        
    except ValidationError as e:
        logger.error(f"Schema validation error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except Exception as e:
        logger.error(f"Error generating schema template: {str(e)}")
        typer.echo(f"ERROR {ERROR_VALIDATION_FAILED}: {str(e)}", err=True)
        raise typer.Exit(code=ERROR_VALIDATION_FAILED)

@app.command()
def list_models(
    log_level: str = typer.Option("INFO", help="Logging level")
):
    """
    List available model types.
    """
    # Update logger with specified log level
    global logger
    logger = setup_logger("synthai_cli", level=log_level)
    
    try:
        models = ModelFactory.list_available_models()
        
        typer.echo("Available models:")
        typer.echo("\nClassification models:")
        for model in models["classification"]:
            typer.echo(f"  - {model}")
        
        typer.echo("\nRegression models:")
        for model in models["regression"]:
            typer.echo(f"  - {model}")
        
        return SUCCESS
        
    except ModelCreationError as e:
        logger.error(f"Model factory error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        typer.echo(f"ERROR {ERROR_MODEL_CREATION_FAILED}: {str(e)}", err=True)
        raise typer.Exit(code=ERROR_MODEL_CREATION_FAILED)

@app.command()
def validate_schema(
    schema_path: str = typer.Argument(..., help="Path to the schema file to validate"),
    log_level: str = typer.Option("INFO", help="Logging level")
):
    """
    Validate a schema file.
    """
    # Update logger with specified log level
    global logger
    logger = setup_logger("synthai_cli", level=log_level)
    
    try:
        validator = SchemaValidator(schema_path)
        schema = validator.load_schema()
        typer.echo(f"Schema at {schema_path} is valid.")
        return SUCCESS
    
    except ValidationError as e:
        logger.error(f"Schema validation error: {e.message}")
        typer.echo(f"ERROR {e.error_code}: {e.message}", err=True)
        raise typer.Exit(code=e.error_code)
    except Exception as e:
        logger.error(f"Error validating schema: {str(e)}")
        typer.echo(f"ERROR {ERROR_VALIDATION_FAILED}: {str(e)}", err=True)
        raise typer.Exit(code=ERROR_VALIDATION_FAILED)

def print_error_codes():
    """
    Print all defined error codes.
    """
    typer.echo("SynthAI Error Codes:")
    for code, message in sorted({k: v for k, v in vars().items() 
                          if k.startswith('ERROR_') or k == 'SUCCESS'}.items(),
                         key=lambda x: x[1]):
        typer.echo(f"  {message:2d}: {get_error_message(message)}")

@app.command()
def error_codes():
    """
    List all error codes used by SynthAI.
    """
    print_error_codes()
    return SUCCESS

if __name__ == "__main__":
    app()