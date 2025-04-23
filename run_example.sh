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
        exit 2
    fi
fi

# Define available examples and models
EXAMPLES=("churn" "promotion" "houses" "reviews")
MODELS=("random_forest" "logistic_regression" "xgboost" "linear_regression")
DEFAULT_OUTPUT_DIR="synthai/models"

# Define compatibility mapping (compatible model types for each example)
# Use simple string-based approach instead of associative arrays for wider shell compatibility
CHURN_MODELS="random_forest logistic_regression xgboost"
PROMOTION_MODELS="random_forest logistic_regression xgboost"
HOUSES_MODELS="random_forest linear_regression xgboost"
REVIEWS_MODELS="random_forest logistic_regression xgboost"

# PnC mode flag (run all datasets with all models)
PNC_MODE=false

# Metadata tracking flag (enabled by default with new schema updates)
METADATA_TRACKING=true

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
            --no-metadata)
                METADATA_TRACKING=false
                shift 1
                ;;
            --help)
                echo -e "${BLUE}Usage: ./run_example.sh [--example <example_name>] [--model <model_type>] [--output <output_dir>] [--all] [--pnc] [--no-metadata]${NC}"
                echo ""
                echo -e "${BLUE}Options:${NC}"
                echo "  --example     Name of example to run"
                echo "               Available examples: churn, promotion, houses, reviews"
                echo "  --model       Type of model to train"
                echo "               Available models: random_forest, logistic_regression, xgboost, linear_regression"
                echo "  --output      Directory to save model (default: synthai/models)"
                echo "  --all         Run all available examples with the specified model"
                echo "  --pnc         Run all compatible example-model combinations"
                echo "  --no-metadata Disable model metadata tracking (not recommended)"
                echo ""
                echo -e "${BLUE}Error Codes:${NC}"
                echo "  1 - Invalid argument or option"
                echo "  2 - Virtual environment not found"
                echo "  3 - Model and dataset are not compatible"
                echo "  4 - Required file not found"
                echo "  5 - Model training process failed"
                echo "  10 - Multiple failures occurred in batch mode"
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Usage: ./run_example.sh [--example <example_name>] [--model <model_type>] [--output <output_dir>] [--all] [--pnc] [--no-metadata]"
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
    echo "6. pnc         - Run all compatible example-model combinations"
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
        echo ""
        
        # Ask the user to choose a model
        echo -e "${CYAN}Which model would you like to use? (Enter 1-4 or the name):${NC}"
        read MODEL_CHOICE
        
        # Process the user's choice
        case $MODEL_CHOICE in
            1|"random_forest") MODEL_TYPE="random_forest" ;;
            2|"logistic_regression") MODEL_TYPE="logistic_regression" ;;
            3|"xgboost") MODEL_TYPE="xgboost" ;;
            4|"linear_regression") MODEL_TYPE="linear_regression" ;;
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
    
    # Ask about metadata tracking
    echo ""
    echo -e "${CYAN}Enable model metadata tracking for enhanced performance analysis? (Y/n):${NC}"
    read METADATA_CHOICE
    
    # Process the user's choice
    if [[ "$METADATA_CHOICE" == "n" || "$METADATA_CHOICE" == "N" ]]; then
        METADATA_TRACKING=false
        echo -e "${YELLOW}Model metadata tracking disabled.${NC}"
    else
        METADATA_TRACKING=true
        echo -e "${GREEN}Model metadata tracking enabled.${NC}"
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to check if a model is compatible with an example
is_model_compatible() {
    local example=$1
    local model=$2
    local compatible_models=""
    
    case $example in
        churn) compatible_models="$CHURN_MODELS" ;;
        promotion) compatible_models="$PROMOTION_MODELS" ;;
        houses) compatible_models="$HOUSES_MODELS" ;;
        reviews) compatible_models="$REVIEWS_MODELS" ;;
        *) return 1 ;;
    esac
    
    # Check if model is in the compatible models list
    for compatible_model in $compatible_models; do
        if [ "$compatible_model" = "$model" ]; then
            return 0  # Model is compatible
        fi
    done
    
    return 1  # Model is not compatible
}

# Function to get compatible models for an example
get_compatible_models() {
    local example=$1
    
    case $example in
        churn) echo "$CHURN_MODELS" ;;
        promotion) echo "$PROMOTION_MODELS" ;;
        houses) echo "$HOUSES_MODELS" ;;
        reviews) echo "$REVIEWS_MODELS" ;;
        *) echo "" ;;
    esac
}

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
    
    # Check if model is compatible with example
    if ! is_model_compatible "$example" "$model_type"; then
        echo -e "${RED}Model type '${model_type}' is not compatible with example '${example}'.${NC}"
        compatible_models=$(get_compatible_models "$example")
        echo -e "${YELLOW}Compatible models for ${example}: ${compatible_models}${NC}"
        return 3
    fi
    
    # Ensure the data file exists
    if [[ ! -f "$DATA_PATH" ]]; then
        echo -e "${RED}Data file not found: $DATA_PATH${NC}"
        return 4
    fi
    
    # Ensure the schema file exists
    if [[ ! -f "$SCHEMA_PATH" ]]; then
        echo -e "${RED}Schema file not found: $SCHEMA_PATH${NC}"
        return 4
    fi
    
    # Configure metadata tracking in the schema if needed
    if [ "$METADATA_TRACKING" = false ]; then
        echo -e "${YELLOW}Disabling metadata tracking in schema${NC}"
        # Create a temporary file
        TMP_FILE="$(mktemp)"
        
        # If model_tracking exists, update it; otherwise add it
        if jq -e '.model_tracking' "$SCHEMA_PATH" > /dev/null; then
            # Update existing model_tracking section
            jq '.model_tracking.enable_metadata = false | .model_tracking.track_performance_history = false' "$SCHEMA_PATH" > "$TMP_FILE"
        else
            # Add new model_tracking section with tracking disabled
            jq '. + {"model_tracking": {"enable_metadata": false, "track_performance_history": false}}' "$SCHEMA_PATH" > "$TMP_FILE"
        fi
        
        # Replace the original file
        cat "$TMP_FILE" > "$SCHEMA_PATH"
        rm "$TMP_FILE"
    else
        echo -e "${GREEN}Model metadata tracking is enabled${NC}"
        
        # Ensure the schema has metadata tracking
        if ! jq -e '.model_tracking' "$SCHEMA_PATH" > /dev/null; then
            echo -e "${YELLOW}Adding metadata tracking to schema${NC}"
            TMP_FILE="$(mktemp)"
            
            # Add model_tracking section with defaults
            jq '. + {"model_tracking": {"enable_metadata": true, "track_performance_history": true, "track_training_metrics": true}}' "$SCHEMA_PATH" > "$TMP_FILE"
            
            # Replace the original file
            cat "$TMP_FILE" > "$SCHEMA_PATH"
            rm "$TMP_FILE"
        fi
    fi
    
    # Special case: Handle XGBoost with house prices regression
    if [ "$example" = "houses" ] && [ "$model_type" = "xgboost" ]; then
        echo -e "${YELLOW}Note: For 'houses' dataset with XGBoost, normalizing target values for better compatibility${NC}"
        # Create a temporary config file with special parameters for XGBoost regression
        XGB_CONFIG_PATH="$(mktemp).json"
        
        if [ "$METADATA_TRACKING" = true ]; then
            echo '{
                "model_params": {
                    "objective": "reg:squarederror",
                    "normalize_target": true
                },
                "training_params": {
                    "epochs": 100,
                    "early_stopping_rounds": 10,
                    "save_metadata": true
                }
            }' > "$XGB_CONFIG_PATH"
        else
            echo '{
                "model_params": {
                    "objective": "reg:squarederror",
                    "normalize_target": true
                }
            }' > "$XGB_CONFIG_PATH"
        fi
        
        CONFIG_ARG="--config $XGB_CONFIG_PATH"
    elif [ "$METADATA_TRACKING" = true ]; then
        # Create a temporary config file with metadata tracking parameters for all models
        META_CONFIG_PATH="$(mktemp).json"
        
        if [ "$model_type" = "xgboost" ]; then
            # For XGBoost models, include epochs
            echo '{
                "training_params": {
                    "epochs": 100,
                    "save_metadata": true,
                    "track_metrics": true
                }
            }' > "$META_CONFIG_PATH"
        else
            # For other models
            echo '{
                "training_params": {
                    "save_metadata": true,
                    "track_metrics": true
                }
            }' > "$META_CONFIG_PATH"
        fi
        
        CONFIG_ARG="--config $META_CONFIG_PATH"
    else
        CONFIG_ARG=""
    fi
    
    # Print the training parameters
    echo -e "${BLUE}=== SynthAI Model Training Example ===${NC}"
    echo -e "${BLUE}Example:     ${NC}${example} (${DESCRIPTION})"
    echo -e "${BLUE}Model type:  ${NC}${model_type}"
    echo -e "${BLUE}Task type:   ${NC}${TASK_TYPE}"
    echo -e "${BLUE}Data file:   ${NC}${DATA_PATH}"
    echo -e "${BLUE}Schema file: ${NC}${SCHEMA_PATH}"
    echo -e "${BLUE}Output dir:  ${NC}${output_dir}"
    echo -e "${BLUE}Metadata:    ${NC}$([ "$METADATA_TRACKING" = true ] && echo "Enabled" || echo "Disabled")"
    echo -e "${BLUE}=======================================${NC}"
    
    # Run the training command - removed the META_ARG which was causing errors
    echo -e "${YELLOW}Starting model training...${NC}"
    if [ -n "$CONFIG_ARG" ]; then
        python -m synthai.cli train "$DATA_PATH" "$SCHEMA_PATH" --model-type "$model_type" --output "$output_dir" $CONFIG_ARG --log-level INFO
    else
        python -m synthai.cli train "$DATA_PATH" "$SCHEMA_PATH" --model-type "$model_type" --output "$output_dir" --log-level INFO
    fi
    
    TRAIN_STATUS=$?
    
    # Clean up temporary files if created
    if [ -n "$CONFIG_ARG" ]; then
        if [ "$example" = "houses" ] && [ "$model_type" = "xgboost" ]; then
            rm -f "$XGB_CONFIG_PATH"
        elif [ "$METADATA_TRACKING" = true ]; then
            rm -f "$META_CONFIG_PATH"
        fi
    fi
    
    # If training succeeded and metadata is enabled, display model metadata
    if [ $TRAIN_STATUS -eq 0 ] && [ "$METADATA_TRACKING" = true ]; then
        echo -e "${YELLOW}Analyzing model metadata...${NC}"
        
        # Get the latest model file
        LATEST_MODEL=$(ls -t "$output_dir"/${model_type}_*.pkl | head -1)
        
        if [ -n "$LATEST_MODEL" ]; then
            # Check if analyze command exists in the CLI
            if python -m synthai.cli --help | grep -q "analyze"; then
                # Use CLI to show model metadata if the analyze command exists
                python -m synthai.cli analyze "$LATEST_MODEL" --show-metadata
                ANALYZE_STATUS=$?
                
                if [ $ANALYZE_STATUS -ne 0 ]; then
                    echo -e "${YELLOW}Note: Model metadata is tracked but the CLI analyze command failed.${NC}"
                    echo -e "${YELLOW}Your model still contains metadata for future use.${NC}"
                fi
            else
                echo -e "${YELLOW}Note: Model contains metadata but the CLI analyze command is not available.${NC}"
                echo -e "${YELLOW}You can access metadata programmatically via the model's get_metadata() method.${NC}"
            fi
        else
            echo -e "${RED}No model file found for analysis.${NC}"
        fi
    fi
    
    # Check if the command was successful
    MODEL_TRAIN_STATUS=$TRAIN_STATUS
    if [ $MODEL_TRAIN_STATUS -eq 0 ]; then
        echo -e "${GREEN}Model training completed successfully!${NC}"
        echo -e "${GREEN}Model saved in: $output_dir${NC}"
        return 0
    else
        echo -e "${RED}Model training failed with status code $MODEL_TRAIN_STATUS${NC}"
        return 5
    fi
}

# Run examples based on the mode
if [ "$PNC_MODE" = true ]; then
    # PnC mode: Run all examples with compatible models
    echo -e "${BLUE}Running all compatible example-model combinations (PnC mode)${NC}"
    
    # Track success/failure across all examples
    TOTAL_COMBOS=0
    FAILURES=0
    FAILED_COMBOS=()
    
    for example in "${EXAMPLES[@]}"; do
        compatible_models=$(get_compatible_models "$example")
        for model in $compatible_models; do
            ((TOTAL_COMBOS++))
            
            echo -e "\n${YELLOW}Running combination: ${example} with ${model}${NC}"
            run_example "$example" "$model" "$OUTPUT_DIR"
            RUN_STATUS=$?
            
            if [ $RUN_STATUS -ne 0 ]; then
                ((FAILURES++))
                FAILED_COMBOS+=("${example}-${model} (Error code: $RUN_STATUS)")
            fi
        done
    done
    
    echo -e "\n${BLUE}=========================================${NC}"
    if [ $FAILURES -eq 0 ]; then
        echo -e "${GREEN}All combinations completed successfully! (${TOTAL_COMBOS}/${TOTAL_COMBOS})${NC}"
        exit 0
    else
        echo -e "${RED}${FAILURES}/${TOTAL_COMBOS} combinations failed:${NC}"
        for failed in "${FAILED_COMBOS[@]}"; do
            echo -e "${RED}- $failed${NC}"
        done
        exit 10
    fi
elif [ "$RUN_ALL" = true ]; then
    # Standard all mode: Run all examples with a single model type
    echo -e "${BLUE}Running all examples with model type: ${MODEL_TYPE}${NC}"
    
    # Track success/failure across all examples
    TOTAL=0
    FAILURES=0
    FAILED_EXAMPLES=()
    
    for example in "${EXAMPLES[@]}"; do
        # Check if model is compatible with example
        if is_model_compatible "$example" "$MODEL_TYPE"; then
            ((TOTAL++))
            run_example "$example" "$MODEL_TYPE" "$OUTPUT_DIR"
            RUN_STATUS=$?
            
            if [ $RUN_STATUS -ne 0 ]; then
                ((FAILURES++))
                FAILED_EXAMPLES+=("${example} (Error code: $RUN_STATUS)")
            fi
        else
            echo -e "\n${YELLOW}Skipping incompatible combination: ${example} with ${MODEL_TYPE}${NC}"
        fi
    done
    
    echo -e "\n${BLUE}=========================================${NC}"
    if [ $FAILURES -eq 0 ]; then
        echo -e "${GREEN}All compatible examples completed successfully! (${TOTAL}/${TOTAL})${NC}"
        exit 0
    else
        echo -e "${RED}${FAILURES}/${TOTAL} examples failed:${NC}"
        for failed in "${FAILED_EXAMPLES[@]}"; do
            echo -e "${RED}- $failed${NC}"
        done
        exit 10
    fi
else
    # Run a single example
    run_example "$EXAMPLE" "$MODEL_TYPE" "$OUTPUT_DIR"
    exit $?
fi