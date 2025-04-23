"""
Command-line interface for the SynthAI Model Training Framework.
This module provides a simple CLI for using the framework.
"""
import os
import logging
import typer
from typing import Optional, List

from synthai.src.pipeline import main as run_pipeline
from synthai.src.utils.logger import setup_logger
from synthai.src.schema.validator import SchemaValidator
from synthai.src.models.model_factory import ModelFactory

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
        import sys
        sys.argv = [
            "pipeline.py",
            f"--data={data}",
            f"--schema={schema}",
            f"--model-type={model_type}",
            f"--output={output}",
            f"--log-level={log_level}"
        ]
        
        if config:
            sys.argv.append(f"--config={config}")
        
        # Run the pipeline
        run_pipeline()
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise typer.Exit(code=1)


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
        
    except Exception as e:
        logger.error(f"Error generating schema template: {str(e)}")
        raise typer.Exit(code=1)


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
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()