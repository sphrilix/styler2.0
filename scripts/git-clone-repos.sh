#!/bin/bash

# Read the repository and commit entries from the CSV file
while IFS=, read -r repository commit; do
    # Clone the repository
    git clone --depth 1 "$repository"

    # Enter the cloned repository directory
    repo_name=$(basename "$repository" .git)
    cd "$repo_name" || exit

    # Checkout the desired commit
    git checkout "$commit"

    # Return to the original directory
    cd ..

# Modify the CSV file name as needed
done < top1javawithCheckstyle-2023-06-07-main.csv
