import logging
import os
import shutil

from src.styler2_0.main import setup_logging

INPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\checkstyled"
OUTPUT_DIR = r"D:\PyCharm_Projects_D\styler2.0\extracted"
NON_VIOLATED = "non_violated"


def get_processed_projects(input_dir: str) -> (list[str], list[str]):
    """
    Separates successfully processed projects from not successfully processed projects.
    :param input_dir: The input directory.
    :return: A tuple of two lists. The first list contains the names of the successfully
     processed projects and the second list contains the names of the not successfully
     processed projects (i.e. the projects that do not have a subfolder NON_VIOLATED).
    """
    successfully_processed = []
    not_successfully_processed = []

    # Iterate over each directory in the input directory
    for directory in os.listdir(input_dir):
        directory = os.path.join(input_dir, directory)
        if os.path.isdir(directory):
            # Check if the subfolder NON_VIOLATED exists
            if not os.path.exists(os.path.join(directory, NON_VIOLATED)):
                not_successfully_processed.append(directory)
            else:
                successfully_processed.append(directory)

    return successfully_processed, not_successfully_processed


def copy_files(input_paths: str, output_dir: str, file_type: str = None) -> None:
    """
    Copies the files from the input paths to the output directory.
    :param input_paths: The paths of the files to copy.
    :param output_dir: The output directory.
    :param file_type: The type of the files to copy. If None, all files are copied.
    """
    # Make a new directory in the output_dir for each path in input_paths
    for path in input_paths:
        # Get the directory name without the path
        dir_name = os.path.basename(path)

        logging.info("Copying files from directory: %s", path)

        # Create a subfolder for each input folder in the output directory
        output_subdir = os.path.join(output_dir, dir_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Iterate over files in the subfolder NON_VIOLATED of the input directory
        for file in os.listdir(os.path.join(path, NON_VIOLATED)):
            file = os.path.join(path, NON_VIOLATED, file)
            # Check if the file is a file and if it has the correct file type
            if os.path.isfile(file) and (file_type is None or file.endswith(file_type)):
                # Copy the file to the output directory
                shutil.copy(file, output_subdir)


def delete_empty_dirs(input_dir: str) -> None:
    """
    Deletes empty directories in the input directory.
    :param input_dir: The input directory.
    :return: None
    """
    for directory in os.listdir(input_dir):
        directory = os.path.join(input_dir, directory)
        if os.path.isdir(directory) and not os.listdir(directory):
            logging.info("Removing empty directory: %s", directory)
            os.rmdir(directory)


def main():
    """
    Extracts all successfully processed files.
    """
    # Setup logging
    log_file = os.path.join(OUTPUT_DIR, "extractor.log")
    setup_logging(log_file)

    # Get the successfully processed and not successfully processed projects
    processed, not_processed = get_processed_projects(input_dir=INPUT_DIR)
    logging.info("Successfully processed: %d \n %s", len(processed), processed)
    logging.info(
        "Not successfully processed: %d \n %s", len(not_processed), not_processed
    )

    # Copy the successfully processed projects to the output directory
    copy_files(input_paths=processed, output_dir=OUTPUT_DIR, file_type=".java")
    logging.info(
        "Copied %d successfully processed dirs to %s.", len(processed), OUTPUT_DIR
    )

    # Remove empty directories
    delete_empty_dirs(OUTPUT_DIR)


if __name__ == "__main__":
    main()
