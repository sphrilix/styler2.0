#!/usr/bin/env python3
import os
import subprocess
import sys


def usage():
    print("Usage:", sys.argv[0], "<input_directory> <output_directory>")
    sys.exit(1)


if len(sys.argv) != 3:
    usage()

input_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
    print(f"Error: Input directory '{input_dir}' does not exist.")
    sys.exit(1)

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the current directory
current_dir = os.getcwd()

# Export PYTHONPATH
os.environ["PYTHONPATH"] = os.pathsep.join(
    [os.environ.get("PYTHONPATH", ""), current_dir]
)

# Iterate over each directory in the "data" directory
for directory in os.listdir(input_dir):
    directory = os.path.join(input_dir, directory)
    if os.path.isdir(directory):
        print("Processing directory:", directory)

        # Get the directory name without the path
        dir_name = os.path.basename(directory)

        # Create a subfolder for each input folder in the "output" directory
        output_subdir = os.path.join(output_dir, dir_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Run the command for each directory separately
        subprocess.run(
            [
                "poetry",
                "run",
                "python",
                "src/styler2_0/main.py",
                "CHECKSTYLE",
                "--save",
                output_subdir,
                "--source",
                directory,
                "--tested",
            ]
        )
