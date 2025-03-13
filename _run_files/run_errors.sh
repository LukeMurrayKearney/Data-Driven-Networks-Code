#!/bin/bash

# Define the first list of strings
datas=("poly" "comix1" "comix2")

# Define the second list of strings
models=("nbinom" "dpln")

# Set the maximum number of parallel jobs
MAX_PROCS=8  # Adjust this number based on your system's capacity

# Function to manage background processes
function parallel_limit {
    # $1: Command to run
    # $2: Arguments for the command

    # Start the command in the background
    "$@" &

    # While the number of background jobs is equal to or greater than MAX_PROCS, wait
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PROCS" ]; do
        sleep 1  # Sleep for a short time before checking again
    done
}

# Loop through each string in datas
for data in "${datas[@]}"; do
    # Loop through each string in models
    for model in "${models[@]}"; do
        # Call the Python script with the current pair of strings in parallel
        parallel_limit python3 _errors.py "$data" "$model"
    done
done

wait