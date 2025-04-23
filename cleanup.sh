#!/bin/bash
# cleanup.sh - Script to clean up generated files

# Set up colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Cleaning up generated files...${NC}"

# Ask for confirmation if no force flag is provided
if [ "$1" != "--force" ]; then
    echo -e "${RED}Warning: This will remove all generated models, logs, and cache files.${NC}"
    read -p "Are you sure you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Cleanup canceled.${NC}"
        exit 0
    fi
fi

# Clean up directories
echo -e "${YELLOW}Removing generated model files...${NC}"
rm -rf synthai/models/*.pkl

echo -e "${YELLOW}Removing processed data...${NC}"
rm -rf synthai/data/processed/*

echo -e "${YELLOW}Removing log files...${NC}"
rm -rf logs/*
rm -rf synthai/logs/*

echo -e "${YELLOW}Removing pytest cache and coverage files...${NC}"
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov

echo -e "${YELLOW}Removing Python cache files...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo -e "${GREEN}Cleanup complete!${NC}"