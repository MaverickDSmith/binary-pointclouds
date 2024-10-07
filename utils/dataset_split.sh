#!/bin/bash

# Usage: ./split_dataset.sh <train_percent> <test_percent> <val_percent> <root_data_dir>

# Arguments
TRAIN_PERCENT=$1
TEST_PERCENT=$2
VAL_PERCENT=$3
ROOT_DIR=$4

# Ensure arguments sum to 100
TOTAL_PERCENT=$((TRAIN_PERCENT + TEST_PERCENT + VAL_PERCENT))
if [ "$TOTAL_PERCENT" -ne 100 ]; then
    echo "Error: Train, test, and validation percentages must sum to 100."
    exit 1
fi

# Ensure ROOT_DIR exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: The specified root data directory does not exist."
    exit 1
fi

# Iterate through directories until we find directories containing files
find "$ROOT_DIR" -type d | while read -r DIR; do
    FILES=($(find "$DIR" -maxdepth 1 -type f)) # Get all files in the current directory
    
    if [ ${#FILES[@]} -gt 0 ]; then
        # Create train, test, and val directories if they don't exist
        mkdir -p "$DIR/train" "$DIR/test" "$DIR/val"

        # Shuffle files randomly
        SHUFFLED_FILES=($(shuf -e "${FILES[@]}"))

        # Calculate number of files for each set
        TOTAL_FILES=${#FILES[@]}
        TRAIN_COUNT=$((TOTAL_FILES * TRAIN_PERCENT / 100))
        TEST_COUNT=$((TOTAL_FILES * TEST_PERCENT / 100))
        VAL_COUNT=$((TOTAL_FILES - TRAIN_COUNT - TEST_COUNT)) # Whatever remains goes to val

        # Distribute files into train, test, and val directories
        for ((i=0; i<$TRAIN_COUNT; i++)); do
            mv "${SHUFFLED_FILES[$i]}" "$DIR/train/"
        done
        for ((i=$TRAIN_COUNT; i<$(($TRAIN_COUNT + $TEST_COUNT)); i++)); do
            mv "${SHUFFLED_FILES[$i]}" "$DIR/test/"
        done
        for ((i=$(($TRAIN_COUNT + $TEST_COUNT)); i<$TOTAL_FILES; i++)); do
            mv "${SHUFFLED_FILES[$i]}" "$DIR/val/"
        done

        echo "Processed directory: $DIR"
        echo "Moved $TRAIN_COUNT files to train, $TEST_COUNT files to test, and $VAL_COUNT files to val."
    fi
done