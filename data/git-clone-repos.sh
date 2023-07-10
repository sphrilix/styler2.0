#!/bin/bash

# Function to process each line in the CSV file
process_csv_line() {
  repository=$1
  commit=$2

  # Trim leading/trailing whitespace from the commit
  commit=$(echo "$commit" | awk '{$1=$1};1')

  # Extract the repository name from the URL
  repo_name=$(basename "$repository" .git)

  # Clone the repository
  git clone --depth 1 "$repository"

  # Enter the cloned repository directory
  cd "$repo_name" || exit

  # Checkout the desired commit
  git checkout "$commit"

  # Return to the original directory
  cd .. || exit
}

# Read the CSV file and process each line
while IFS=, read -r repository commit; do
  process_csv_line "$repository" "$commit"

# Modify the CSV file name as needed
done < repos.csv
