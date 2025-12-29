# MLtrainer UI

This is the web interface for the SynthAI Model Training Framework.

## Features

- **Train Models**: Configure and start training jobs via a user-friendly form.
- **Manage Models**: View trained models and their metadata.
- **Inference**: Serve models and run predictions directly from the browser.

## Getting Started

1.  **Prerequisites**: Ensure the `synthai` package is installed and accessible.
    ```bash
    # In the parent directory
    source ../venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    npm install
    # or
    yarn
    ```

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```

4.  **Open Browser**: Navigate to [http://localhost:3000](http://localhost:3000).

## Architecture

The UI interacts with the SynthAI CLI via Next.js API routes.
- `/api/train`: Triggers `python -m synthai.cli train`
- `/api/models`: Lists files in the `models/` directory
- `/api/predict`: Interacts with the `serve` command or runs inference script

