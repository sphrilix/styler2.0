TEST_SUFFIXES = ["test", "Test", "_test", "_Test"]


class Tested:
    """
    A class that checks if a file is tested against a set of test files.
    """

    def __init__(self, test_files):
        """
        Creates a new Tested object.
        @param test_files: The test files to check against.
        """
        self.test_files = set()
        for file in test_files:
            self.test_files.add(file.name.split(".")[0])

    def is_tested(self, src_file) -> bool:
        """
        Returns True if the file is tested, False otherwise.
        Files are assumed to be tested if there exists a file with the same name
        with a test suffix.
        @param src_file: The file to check.
        @return: True if the file is tested, False otherwise.
        """
        name_without_suffix = src_file.name.split(".")[0]
        possible_test_filenames = [
            name_without_suffix + suffix for suffix in TEST_SUFFIXES
        ]

        for possible_test_filename in possible_test_filenames:
            if possible_test_filename in self.test_files:
                return True

        return False


def split_test_files(files):
    """
    Splits the files into src and test files.
    A file is considered a test file if it has a test suffix.
    @param files: The files to split.
    @return: A tuple of (src_files, test_files).
    """
    src_files = set()
    test_files = set()

    for file in files:
        name_without_suffix = file.name.split(".")[0]
        if name_without_suffix.endswith(tuple(TEST_SUFFIXES)):
            test_files.add(file)
        else:
            src_files.add(file)

    return src_files, test_files
