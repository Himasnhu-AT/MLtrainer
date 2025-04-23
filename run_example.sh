#!/bin/bash
# run_example.sh - Script to run example model training for SynthAI

# Set up colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure the virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found. Please run ./setup.sh first.${NC}"
        exit 1
    fi
fi

# Default values
EXAMPLE="churn"  # Default example to run
MODEL_TYPE="random_forest"  # Default model type
OUTPUT_DIR="synthai/models"

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --example)
            EXAMPLE="$2"
            shift 2
            ;;
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_example.sh [--example <example_name>] [--model <model_type>] [--output <output_dir>]"
            echo ""
            echo "Options:"
            echo "  --example  Name of example to run (default: churn)"
            echo "             Available examples: churn"
            echo "  --model    Type of model to train (default: random_forest)"
            echo "             Available models: random_forest, logistic_regression, xgboost"
            echo "  --output   Directory to save model (default: synthai/models)"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./run_example.sh [--example <example_name>] [--model <model_type>] [--output <output_dir>]"
            echo "Use --help for more information."
            exit 1
            ;;
    esac
done

# Check if the example is valid
if [[ "$EXAMPLE" != "churn" ]]; then
    echo -e "${RED}Invalid example: $EXAMPLE${NC}"
    echo "Available examples: churn"
    exit 1
fi

# Set up parameters based on the chosen example
if [[ "$EXAMPLE" == "churn" ]]; then
    DATA_PATH="synthai/data/raw/customer_churn.csv"
    SCHEMA_PATH="synthai/config/customer_churn_schema.json"
    DESCRIPTION="Customer churn prediction"
fi

# Ensure the data file exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo -e "${RED}Data file not found: $DATA_PATH${NC}"
    exit 1
fi

# Ensure the schema file exists
if [[ ! -f "$SCHEMA_PATH" ]]; then
    echo -e "${RED}Schema file not found: $SCHEMA_PATH${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print the training parameters
echo -e "${BLUE}=== SynthAI Model Training Example ===${NC}"
echo -e "${BLUE}Example:     ${NC}${EXAMPLE} (${DESCRIPTION})"
echo -e "${BLUE}Model type:  ${NC}${MODEL_TYPE}"
echo -e "${BLUE}Data file:   ${NC}${DATA_PATH}"
echo -e "${BLUE}Schema file: ${NC}${SCHEMA_PATH}"
echo -e "${BLUE}Output dir:  ${NC}${OUTPUT_DIR}"
echo -e "${BLUE}=======================================${NC}"

# Run the training command
echo -e "${YELLOW}Starting model training...${NC}"
python -m synthai.cli train "$DATA_PATH" "$SCHEMA_PATH" --model-type "$MODEL_TYPE" --output "$OUTPUT_DIR"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Model training completed successfully!${NC}"
    echo -e "${GREEN}Model saved in: $OUTPUT_DIR${NC}"
    exit 0
else
    echo -e "${RED}Model training failed.${NC}"
    exit 1
fi