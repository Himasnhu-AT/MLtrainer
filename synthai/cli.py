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
    log_level: str = typer.Option("INFO", help="Logging level")
):
    """
    Train a machine learning model using the specified data and schema.
    """
    try:
        # Set up arguments for the pipeline
        logger.info(f"Training model with data: {data}, schema: {schema}, model type: {model_type}")
        logger.info(f"Output directory: {output}, log level: {log_level}")
        
        # Replace sys.argv with our actual arguments
        sys.argv = [
            "pipeline.py",
            "--data", data,
            "--schema", schema,
            "--model-type", model_type,
            "--output", output,
            "--log-level", log_level
        ]
        
        if config:
            sys.argv.extend(["--config", config])
        
        # Run the pipeline
        logger.info("Starting the training pipeline")
        run_pipeline()
        logger.info("Training completed successfully")
        return SUCCESS
        
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
):
    """
    Generate a sample schema template.
    """
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
def list_models():
    """
    List available model types.
    """
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
    schema_path: str = typer.Argument(..., help="Path to the schema file to validate")
):
    """
    Validate a schema file.
    """
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