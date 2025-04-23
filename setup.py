"""
Setup script for the SynthAI Model Training Framework.
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="synthai",
    version="0.1.0",
    author="SynthAI Team",
    author_email="team@synthai.ai",
    description="A framework for training ML models from CSV data using JSON schema",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/synthai/model_training",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "pydantic>=1.9.0",
        "jsonschema>=4.0.0",
        "typer>=0.4.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "mlflow>=1.20.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0"
    ],
    entry_points={
        "console_scripts": [
            "synthai=synthai.cli:app",
        ],
    }
)