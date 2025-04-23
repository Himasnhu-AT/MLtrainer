# SynthAI Model Training Framework - Architecture

This document outlines the architecture of the SynthAI Model Training Framework, a system designed to simplify machine learning model training using structured data definitions.

## System Overview

The SynthAI Model Training Framework is designed to automate the process of training machine learning models from structured data (primarily CSV files) using a schema-driven approach. By defining a JSON schema for your data, the framework can:

1. Validate input data against the schema
2. Apply appropriate preprocessing techniques based on data types
3. Train a suitable model
4. Evaluate the model's performance
5. Save the model and preprocessing pipeline for later use

## Component Architecture

The framework uses a modular architecture with several key components:

```
                                  ┌─────────────────┐
                                  │                 │
                                  │   CLI / API     │
                                  │                 │
                                  └────────┬────────┘
                                           │
                                           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│ Schema Validator│◄───┤    Pipeline     ├───►│   Data Loader   │
│                 │    │                 │    │                 │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Model Factory  │◄───┤  Preprocessor   ├───►│Model Evaluator  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **Schema Validator**
   - Validates JSON schema definitions
   - Ensures data conforms to the defined schema
   - Provides templates for schema creation

2. **Data Loader**
   - Loads data from CSV and other formats
   - Handles basic data inspection and summary
   - Provides sampling capabilities

3. **Data Preprocessor**
   - Applies schema-defined preprocessing to features
   - Handles missing values, scaling, encoding
   - Manages feature transformations (numerical, categorical, text)
   - Handles train/test splitting

4. **Model Factory**
   - Creates appropriate model instances
   - Supports classification and regression models
   - Configurable through model parameters

5. **Model Evaluator**
   - Calculates performance metrics
   - Supports cross-validation
   - Provides detailed evaluation reports

6. **Pipeline**
   - Orchestrates the entire training process
   - Handles logging and error management
   - Saves model artifacts

7. **CLI**
   - Provides a command-line interface
   - Exposes core functionality to users

## Data Flow

1. User defines a JSON schema describing their data structure
2. User provides a CSV file containing the data
3. The Pipeline reads both inputs and initiates the process
4. Schema Validator validates the schema and checks data compliance
5. Data Loader loads the data into memory
6. Preprocessor transforms the data according to schema directives
7. Model Factory creates the appropriate model
8. Model is trained on the preprocessed data
9. Model Evaluator assesses model performance
10. Trained model and preprocessor are saved for later use

## Schema Format

The schema is a JSON document that defines:

- **Features**: Data columns with their types and preprocessing directives
- **Target**: The target variable to predict
- **Metadata**: Additional information about the dataset and model

Example schema:

```json
{
  "features": [
    {"name": "age", "type": "numeric", "preprocessing": "scale"},
    {"name": "income", "type": "numeric", "preprocessing": "minmax"},
    {"name": "occupation", "type": "categorical", "preprocessing": "one-hot"}
  ],
  "target": {"name": "churn", "type": "binary"},
  "metadata": {
    "description": "Customer churn prediction model",
    "version": "1.0"
  }
}
```

## Extension Points

The framework is designed to be extensible in several ways:

1. **New Feature Types**: Add support for new data types by extending the preprocessor
2. **New Preprocessing Methods**: Implement additional preprocessing techniques
3. **Custom Models**: Add new model types to the model factory
4. **Custom Metrics**: Define new evaluation metrics in the evaluator
5. **Output Formats**: Extend model saving and export capabilities

## Deployment

The framework can be deployed in different ways:

1. **Standalone Tool**: Installed via pip and used from the command line
2. **Library**: Imported and used programmatically in Python applications
3. **Service**: Wrapped in a web service for remote model training

## Future Directions

Planned features for future development:

1. **Hyperparameter Optimization**: Automatic tuning of model parameters
2. **Feature Selection**: Automatic selection of most relevant features
3. **Model Explainability**: Tools for understanding model decisions
4. **Drift Detection**: Monitoring data and model drift over time
5. **Pipeline Versioning**: Tracking changes to model training pipelines