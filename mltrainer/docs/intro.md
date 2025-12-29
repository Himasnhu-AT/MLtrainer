# MLtrainer UI Documentation

## Overview

The MLtrainer UI provides a graphical interface for the SynthAI framework. It allows users to train, manage, and serve machine learning models without needing to use the command line directly.

## Components

### Training Page (`/train`)
- **Form**: Allows users to specify:
    - Data Path (CSV)
    - Schema Path (JSON)
    - Model Type (Random Forest, XGBoost, etc.)
    - Output Directory
    - Hyperparameter Tuning (Toggle)
- **Action**: Submits a job to the backend which runs the SynthAI pipeline.

### Models Page (`/models`)
- **List**: Shows all trained models in the configured output directory.
- **Actions**:
    - **Serve**: Starts a REST API server for the selected model.
    - **Predict**: Opens a form to send data to the served model and view results.

## Backend Integration

The Next.js API routes act as a bridge between the frontend and the Python backend.
- It uses `child_process.exec` or `spawn` to run Python commands.
- It assumes the Python environment is set up correctly (virtual environment activated or packages installed globally).
