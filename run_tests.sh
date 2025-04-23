#!/bin/bash
# run_tests.sh - Script to run all or specific tests for SynthAI

# Set up colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Parse command line arguments
TEST_PATH="synthai/tests"  # Default: run all tests
VERBOSE="-v"              # Default: verbose output
COVERAGE=""               # Default: no coverage report

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_PATH="synthai/tests/unit"
            shift
            ;;
        --file)
            if [[ -n "$2" && -f "$2" ]]; then
                TEST_PATH="$2"
                shift 2
            else
                echo -e "${RED}Please provide a valid test file path after --file${NC}"
                exit 1
            fi
            ;;
        --quiet)
            VERBOSE=""
            shift
            ;;
        --coverage)
            COVERAGE="--cov=synthai --cov-report=term --cov-report=html"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./run_tests.sh [--unit] [--file <test_file>] [--quiet] [--coverage]"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Running tests in: ${TEST_PATH}${NC}"

# Run pytest with the specified arguments
if [[ -n "$COVERAGE" ]]; then
    echo -e "${YELLOW}Running tests with coverage...${NC}"
    python -m pytest ${TEST_PATH} ${VERBOSE} ${COVERAGE}
else
    python -m pytest ${TEST_PATH} ${VERBOSE}
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    if [[ -n "$COVERAGE" ]]; then
        echo -e "${YELLOW}Coverage report generated in htmlcov/index.html${NC}"
    fi
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi