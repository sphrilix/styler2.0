#!/bin/bash

# Function to display usage information
function usage() {
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
}

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    usage
fi

# Store the input and output directories provided as arguments
input_dir="$1"
output_dir="$2"

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' does not exist."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Save the current directory
current_dir="$PWD"

# Export PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${current_dir}"

# Iterate over each directory in the "data" directory
for directory in "$input_dir"/*; do
  if [ -d "$directory" ]; then
    echo "Processing directory: $directory"

    # Get the directory name without the path
    dir_name=$(basename "$directory")

    # Create a subfolder for each input folder in the "output" directory
    output_subdir="$output_dir/${dir_name}"
    mkdir -p "$output_subdir"

    # Run the command for each directory separately
    poetry run python src/styler2_0/main.py CHECKSTYLE --save "$output_subdir" --source "$directory" --tested
  fi
done
