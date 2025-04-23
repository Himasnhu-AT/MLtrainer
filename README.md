# SynthAI Model Training Framework

A robust and flexible framework for training machine learning models from CSV data using a JSON schema definition.

## Overview

SynthAI is a model training framework that simplifies the process of training machine learning models. It takes structured data in CSV format along with a JSON schema definition and produces trained models ready for deployment.

## Features

- **Schema-driven processing**: Define your data structure through a JSON schema
- **Automated preprocessing**: Handles data cleaning, normalization, and feature engineering
- **Flexible model selection**: Choose from various ML algorithms or bring your own
- **Validation and evaluation**: Built-in cross-validation and performance metrics
- **Model persistence**: Export trained models in various formats
- **Extensible pipeline**: Easy to customize and extend for specific use cases

## Project Structure

```
synthai/
├── src/                # Source code
│   ├── data/           # Data processing modules
│   ├── models/         # Model definition and training
│   ├── schema/         # Schema validation and handling
│   ├── utils/          # Utility functions
│   └── pipeline.py     # Main pipeline orchestration
├── tests/              # Test files
├── models/             # Saved model artifacts
├── data/               # Data directory
│   ├── raw/            # Raw input data
│   └── processed/      # Processed data
├── config/             # Configuration files
└── docs/               # Documentation
```

## Installation

### Using Bash Scripts (Recommended)

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script to create a virtual environment and install dependencies
./setup.sh
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthai.git
cd synthai

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using Bash Scripts

```bash
# List available models
./list_models.sh

# Run the customer churn prediction example with default settings (random forest)
./run_example.sh

# Run with a specific model type
./run_example.sh --model xgboost

# See all available options
./run_example.sh --help
```

### Manual Execution

1. Define your data schema in a JSON file:

```json
{
  "features": [
    {"name": "feature1", "type": "numeric", "preprocessing": "scale"},
    {"name": "feature2", "type": "categorical", "preprocessing": "one-hot"},
    {"name": "feature3", "type": "text", "preprocessing": "tfidf"}
  ],
  "target": {"name": "target_column", "type": "binary"}
}
```

2. Place your CSV data in the `data/raw/` directory

3. Run the training pipeline:

```bash
python -m synthai.cli train --data synthai/data/raw/your_data.csv --schema synthai/config/your_schema.json --model-type "random_forest"
```

4. Find your trained model in the `models/` directory

## Configuration Options

You can customize the training process through command-line arguments or a configuration file:

- `--data`: Path to input CSV file
- `--schema`: Path to JSON schema definition
- `--model-type`: Type of model to train (e.g., random_forest, xgboost, neural_network)
- `--output`: Directory for saving the model
- `--config`: Path to a YAML configuration file with additional options

## Development

### Prerequisites

- Python 3.8+
- Required packages: See `requirements.txt`

### Running Tests

```bash
# Using the test script (recommended)
chmod +x run_tests.sh

# Run all tests
./run_tests.sh

# Run only unit tests
./run_tests.sh --unit

# Run with coverage report
./run_tests.sh --coverage

# Run a specific test file
./run_tests.sh --file synthai/tests/unit/test_schema_validator.py
```

### Manual Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_schema_validator.py
```

### Cleanup

```bash
# Clean up generated files, models, and caches
chmod +x cleanup.sh
./cleanup.sh

# Force cleanup without confirmation
./cleanup.sh --force
```

## Utility Scripts

The following bash scripts are provided to simplify common tasks:

- `setup.sh` - Creates a virtual environment and installs dependencies
- `run_example.sh` - Runs example model training with configurable options
- `run_tests.sh` - Runs tests with various options (unit tests, coverage, etc.)
- `list_models.sh` - Lists available model types in the framework
- `cleanup.sh` - Removes generated files, models, and cache files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.