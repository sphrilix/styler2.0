TEST_SUFFIXES = ["test", "Test", "_test", "_Test"]


def _split_test_files(files):
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


def extract_tested_src_files(files) -> set:
    """
    Returns the set of tested src files.
    Files are assumed to be a src file if they do not have a test suffix.
    Files are assumed to be tested if there exists a file with the same name with a test
     suffix.
    @param files: The files to check.
    """
    # split files into src and test files
    src_files, test_files = _split_test_files(files)

    # create the actual test filenames
    test_file_names = _test_filenames(test_files)

    # return the set of tested src files
    return _tested_files(src_files, test_file_names)


def _tested_files(src_files, test_file_names):
    """
    Returns the set of tested src files.
    @param src_files: The src files.
    @param test_file_names: The actual test file names.
    """
    tested_files = set()
    for src_file in src_files:
        name_without_suffix = src_file.name.split(".")[0]
        possible_test_filenames = {
            name_without_suffix + suffix for suffix in TEST_SUFFIXES
        }

        if possible_test_filenames.intersection(test_file_names):
            tested_files.add(src_file)

    return tested_files


def _test_filenames(test_files):
    """
    Returns the set of test filenames.
    @param test_files: The test files.
    """
    test_file_names = set()
    for file in test_files:
        test_file_names.add(file.name.split(".")[0])
    return test_file_names
