#!/bin/bash
# list_models.sh - Script to list available models in SynthAI

# Set up colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo -e "${BLUE}=== SynthAI Available Models ===${NC}"

# Directly import and run the command from the cli module to avoid any issues with command discovery
python -c "from synthai.cli import app; app(['list-models'])" || {
    # If the above fails, try the module approach
    echo -e "${YELLOW}Trying alternative method...${NC}"
    python -m synthai.cli list-models
}

echo -e "\n${YELLOW}To generate a sample schema, use:${NC}"
echo -e "python -m synthai.cli generate-schema <output_path>${NC}"