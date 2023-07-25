#!/bin/bash

# Function to process each line in the CSV file
process_csv_line() {
  repository=$1
  commit=$2

  # Trim leading/trailing whitespace from the commit
  commit=$(echo "$commit" | awk '{$1=$1};1')

  # Extract the repository name from the URL
  repo_name=$(basename "$repository" .git)

  # Check if the repository directory already exists
  if [ -d "$repo_name" ]; then
    echo "Repository '$repo_name' already exists. Skipping clone and checkout."
  else
    # Clone the repository
    git clone --depth 1 "$repository"

    # Enter the cloned repository directory
    cd "$repo_name" || exit

    # Checkout the desired commit
    git checkout "$commit"

    # Return to the original directory
    cd .. || exit
  fi
}

# Check if the CSV file argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <csv_file>"
  exit 1
fi

csv_file="$1"

# Check if the CSV file exists
if [ ! -f "$csv_file" ]; then
  echo "CSV file not found: $csv_file"
  exit 1
fi

# Read the CSV file and process each line
while IFS=, read -r repository commit; do
  process_csv_line "$repository" "$commit"
done < "$csv_file"
