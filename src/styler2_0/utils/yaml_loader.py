import os

import yaml

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
RESOURCES_DIR = os.path.join(CURR_DIR, "../../res")


def load(file_name: str, file_path: str = RESOURCES_DIR) -> dict:
    """
    Loads the given yaml file.
    :param file_name: The given file name.
    :param file_path: The given file path.
    :return: Returns the loaded yaml file as dict.
    """
    with open(os.path.join(file_path, file_name), encoding="utf-8") as file_stream:
        return yaml.safe_load(file_stream)
