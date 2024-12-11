#!/bin/bash

# Loop through each config number from 1 to 12
for i in {1..12}
do
    echo "Running training with config $i"
    python3 train.py --config "./experiments/experiment1/configs/config$i.yaml"
    
    # Check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Training failed on config $i. Exiting."
        exit 1
    fi
done

echo "All training runs completed successfully."