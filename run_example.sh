#!/bin/bash
# run_example.sh - Script to run example model training for SynthAI

# Set up colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

# Define available examples and models
EXAMPLES=("churn" "promotion" "houses" "reviews")
MODELS=("random_forest" "logistic_regression" "xgboost" "linear_regression" "decision_tree")
DEFAULT_OUTPUT_DIR="synthai/models"

# PnC mode flag (run all datasets with all models)
PNC_MODE=false

# Check if any arguments were provided
if [ $# -gt 0 ]; then
    # Process command line arguments
    EXAMPLE=""
    MODEL_TYPE=""
    OUTPUT_DIR=""
    RUN_ALL=false

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
            --all)
                RUN_ALL=true
                shift 1
                ;;
            --pnc)
                PNC_MODE=true
                RUN_ALL=true
                shift 1
                ;;
            --help)
                echo -e "${BLUE}Usage: ./run_example.sh [--example <example_name>] [--model <model_type>] [--output <output_dir>] [--all] [--pnc]${NC}"
                echo ""
                echo -e "${BLUE}Options:${NC}"
                echo "  --example  Name of example to run"
                echo "             Available examples: churn, promotion, houses, reviews"
                echo "  --model    Type of model to train"
                echo "             Available models: random_forest, logistic_regression, xgboost, linear_regression, decision_tree"
                echo "  --output   Directory to save model (default: synthai/models)"
                echo "  --all      Run all available examples with the specified model"
                echo "  --pnc      Run all examples with all model types (PnC mode)"
                echo ""
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Usage: ./run_example.sh [--example <example_name>] [--model <model_type>] [--output <output_dir>] [--all] [--pnc]"
                echo "Use --help for more information."
                exit 1
                ;;
        esac
    done

    # Set defaults if not provided
    [ -z "$EXAMPLE" ] && [ "$RUN_ALL" = false ] && EXAMPLE="churn"
    [ -z "$MODEL_TYPE" ] && MODEL_TYPE="random_forest"
    [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
else
    # Interactive mode - no arguments provided
    # Present available options to the user
    
    echo -e "${BLUE}=== SynthAI Model Training Examples ===${NC}"
    echo -e "${YELLOW}This script will help you train machine learning models using SynthAI.${NC}"
    echo ""

    # Display available examples
    echo -e "${BLUE}Available Example Datasets:${NC}"
    echo "1. churn       - Customer churn prediction (binary classification)"
    echo "2. promotion   - Employee promotion prediction (binary classification)"
    echo "3. houses      - House price prediction (regression)"
    echo "4. reviews     - Product review sentiment analysis (multiclass classification)"
    echo "5. all         - Run all examples with a single model type"
    echo "6. pnc         - Run all examples with all model types (PnC mode)"
    echo ""

    # Ask the user to choose an example
    echo -e "${CYAN}Which example would you like to run? (Enter 1-6 or the name):${NC}"
    read EXAMPLE_CHOICE
    
    # Process the user's choice
    RUN_ALL=false
    PNC_MODE=false
    case $EXAMPLE_CHOICE in
        1|"churn") EXAMPLE="churn" ;;
        2|"promotion") EXAMPLE="promotion" ;;
        3|"houses") EXAMPLE="houses" ;;
        4|"reviews") EXAMPLE="reviews" ;;
        5|"all") RUN_ALL=true ;;
        6|"pnc") RUN_ALL=true; PNC_MODE=true ;;
        *)
            echo -e "${RED}Invalid choice. Using default example (churn).${NC}"
            EXAMPLE="churn"
            ;;
    esac
    
    # Only ask for model type if not in PnC mode
    if [ "$PNC_MODE" = false ]; then
        # Display available models
        echo ""
        echo -e "${BLUE}Available Model Types:${NC}"
        echo "1. random_forest       - Random Forest (classification & regression)"
        echo "2. logistic_regression - Logistic Regression (classification only)"
        echo "3. xgboost             - XGBoost (classification & regression)"
        echo "4. linear_regression   - Linear Regression (regression only)"
        echo "5. decision_tree       - Decision Tree (classification & regression)"
        echo ""
        
        # Ask the user to choose a model
        echo -e "${CYAN}Which model would you like to use? (Enter 1-5 or the name):${NC}"
        read MODEL_CHOICE
        
        # Process the user's choice
        case $MODEL_CHOICE in
            1|"random_forest") MODEL_TYPE="random_forest" ;;
            2|"logistic_regression") MODEL_TYPE="logistic_regression" ;;
            3|"xgboost") MODEL_TYPE="xgboost" ;;
            4|"linear_regression") MODEL_TYPE="linear_regression" ;;
            5|"decision_tree") MODEL_TYPE="decision_tree" ;;
            *)
                echo -e "${RED}Invalid choice. Using default model (random_forest).${NC}"
                MODEL_TYPE="random_forest"
                ;;
        esac
    fi
    
    # Ask for output directory
    echo ""
    echo -e "${CYAN}Where would you like to save the model? (Press Enter for default: $DEFAULT_OUTPUT_DIR):${NC}"
    read OUTPUT_DIR_CHOICE
    
    # Process the user's choice
    if [ -z "$OUTPUT_DIR_CHOICE" ]; then
        OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    else
        OUTPUT_DIR="$OUTPUT_DIR_CHOICE"
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to run a single example
run_example() {
    local example=$1
    local model_type=$2
    local output_dir=$3
    
    echo -e "\n${BLUE}=========================================${NC}"
    
    # Set up parameters based on the chosen example
    case $example in
        churn)
            DATA_PATH="synthai/data/raw/customer_churn.csv"
            SCHEMA_PATH="synthai/config/customer_churn_schema.json"
            DESCRIPTION="Customer churn prediction"
            TASK_TYPE="classification"
            ;;
        promotion)
            DATA_PATH="synthai/data/raw/employee_promotion.csv"
            SCHEMA_PATH="synthai/config/employee_promotion_schema.json"
            DESCRIPTION="Employee promotion prediction"
            TASK_TYPE="classification"
            ;;
        houses)
            DATA_PATH="synthai/data/raw/house_prices.csv"
            SCHEMA_PATH="synthai/config/house_prices_schema.json"
            DESCRIPTION="House price prediction"
            TASK_TYPE="regression"
            # For regression tasks, adapt model if needed
            if [[ "$model_type" == "logistic_regression" ]]; then
                echo -e "${YELLOW}Note: Switching to linear_regression for regression task${NC}"
                model_type="linear_regression"
            fi
            ;;
        reviews)
            DATA_PATH="synthai/data/raw/product_reviews.csv"
            SCHEMA_PATH="synthai/config/product_reviews_schema.json"
            DESCRIPTION="Product review sentiment analysis"
            TASK_TYPE="classification"
            ;;
        *)
            echo -e "${RED}Invalid example: $example${NC}"
            echo "Available examples: churn, promotion, houses, reviews"
            return 1
            ;;
    esac
    
    # Ensure the data file exists
    if [[ ! -f "$DATA_PATH" ]]; then
        echo -e "${RED}Data file not found: $DATA_PATH${NC}"
        return 1
    fi
    
    # Ensure the schema file exists
    if [[ ! -f "$SCHEMA_PATH" ]]; then
        echo -e "${RED}Schema file not found: $SCHEMA_PATH${NC}"
        return 1
    fi
    
    # Print the training parameters
    echo -e "${BLUE}=== SynthAI Model Training Example ===${NC}"
    echo -e "${BLUE}Example:     ${NC}${example} (${DESCRIPTION})"
    echo -e "${BLUE}Model type:  ${NC}${model_type}"
    echo -e "${BLUE}Data file:   ${NC}${DATA_PATH}"
    echo -e "${BLUE}Schema file: ${NC}${SCHEMA_PATH}"
    echo -e "${BLUE}Output dir:  ${NC}${output_dir}"
    echo -e "${BLUE}=======================================${NC}"
    
    # Run the training command
    echo -e "${YELLOW}Starting model training...${NC}"
    python -m synthai.cli train "$DATA_PATH" "$SCHEMA_PATH" --model-type "$model_type" --output "$output_dir"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model training completed successfully!${NC}"
        echo -e "${GREEN}Model saved in: $output_dir${NC}"
        return 0
    else
        echo -e "${RED}Model training failed.${NC}"
        return 1
    fi
}

# Run examples based on the mode
if [ "$PNC_MODE" = true ]; then
    # PnC mode: Run all examples with all models
    echo -e "${BLUE}Running all examples with all model types (PnC mode)${NC}"
    
    # Track success/failure across all examples
    TOTAL_COMBOS=0
    FAILURES=0
    
    for example in "${EXAMPLES[@]}"; do
        # Get task type for this example
        case $example in
            houses) TASK_TYPE="regression" ;;
            *) TASK_TYPE="classification" ;;
        esac
        
        for model in "${MODELS[@]}"; do
            # Skip incompatible combinations
            if [[ "$TASK_TYPE" == "regression" && "$model" == "logistic_regression" ]]; then
                echo -e "${YELLOW}Skipping incompatible combination: $example with $model${NC}"
                continue
            fi
            
            ((TOTAL_COMBOS++))
            run_example "$example" "$model" "$OUTPUT_DIR"
            if [ $? -ne 0 ]; then
                ((FAILURES++))
            fi
        done
    done
    
    echo -e "\n${BLUE}=========================================${NC}"
    if [ $FAILURES -eq 0 ]; then
        echo -e "${GREEN}All combinations completed successfully! (${TOTAL_COMBOS}/${TOTAL_COMBOS})${NC}"
        exit 0
    else
        echo -e "${RED}${FAILURES}/${TOTAL_COMBOS} combinations failed.${NC}"
        exit 1
    fi
elif [ "$RUN_ALL" = true ]; then
    # Standard all mode: Run all examples with a single model type
    echo -e "${BLUE}Running all examples with model type: ${MODEL_TYPE}${NC}"
    
    # Track success/failure across all examples
    FAILURES=0
    
    for example in "${EXAMPLES[@]}"; do
        run_example "$example" "$MODEL_TYPE" "$OUTPUT_DIR"
        if [ $? -ne 0 ]; then
            ((FAILURES++))
        fi
    done
    
    echo -e "\n${BLUE}=========================================${NC}"
    if [ $FAILURES -eq 0 ]; then
        echo -e "${GREEN}All examples completed successfully!${NC}"
        exit 0
    else
        echo -e "${RED}${FAILURES} example(s) failed.${NC}"
        exit 1
    fi
else
    # Run a single example
    run_example "$EXAMPLE" "$MODEL_TYPE" "$OUTPUT_DIR"
    exit $?
fi