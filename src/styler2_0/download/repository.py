import csv
import json

import requests

API_URL = "https://api.github.com"
SEARCH = "search"
COMMITS = "commits"
CODE = "code"
REPOS = "repos"

# TODO: Use environment variable and add new token
AUTH_TOKEN = "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"
MAX_INT = 1000000
HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}


# TODO: Work with python class instead of dict?


class DownloadFailedException(Exception):
    """
    Exception that is thrown whenever the download failed.
    """


def get_checkstyle_repos(
    per_page: int = 100, first_page: int = 1, last_page: int = 1
) -> dict[str, dict[str, str]]:
    # Define the API endpoint and query parameters
    api_url = API_URL + "/" + SEARCH + "/" + CODE
    query_params = {
        "q": "checkstyle filename:pom.xml",
        "per_page": per_page,
        "page": first_page,
    }

    data = {}

    while True:
        # Make the GET request
        response = requests.get(api_url, params=query_params, headers=HEADERS)

        # Check the response status code
        if response.status_code == 200:
            # Convert the response json to a python dictionary and add it to the data
            search_response_json = response.json()
            for item in search_response_json["items"]:
                repository = item["repository"]
                repository_name = repository["full_name"]
                repository_url = repository["url"]

                # Open the repository url and save the json
                repository_response = requests.get(repository_url, headers=HEADERS)
                if repository_response.status_code == 200:
                    repository_response_json = repository_response.json()
                    data[repository_name] = repository_response_json
                else:
                    raise DownloadFailedException("Download failed.")

            # Check if there are more pages to fetch
            if "next" in response.links and query_params["page"] < last_page:
                # Update the query parameters to fetch the next page
                query_params["page"] += 1
            else:
                # No more pages, exit the loop
                break
        else:
            raise DownloadFailedException("Download failed.")
    return data


def remove_keys(
    data: dict[str, dict[str, str]], keys: list[str]
) -> dict[str, dict[str, str]]:
    for key in keys:
        for item in data.values():
            item.pop(key, None)
    return data


class RepositorySortCriteria:
    """
    Class that represents the criteria to sort the repositories.
    The sort criteria is a list of tuples (key, weight).
    TODO: Add ASC/DESC
    """

    def __init__(self, criteria=None) -> None:
        """
        Initializes the sort criteria.
        :param criteria: The sort criteria.
        """
        if criteria is None:
            criteria = {}
        self.criteria = criteria

    def add(self, key: str, weight: int) -> None:
        """
        Adds a criterion to the sort criteria.
        :param key: The key to sort by.
        :param weight: The weight of the criterion.
        """
        self.criteria[key] = weight

    def custom_sort_key(self, item):
        return sum(
            self.criteria[key] * item[key] for key in item if key in self.criteria
        )


class RepositoryFilterCriteria:
    """
    Class that represents the criteria to filter the repositories.
    The filter criteria is a list of triples of:
    - (key, min, max) if the key is a number.
    - (key, value) if the key is a boolean.
    """

    def __init__(self, criteria=None) -> None:
        """
        Initializes the filter criteria.
        :param criteria: The filter criteria.
        """
        if criteria is None:
            criteria = {}
        self.criteria = criteria

    def add_bool(self, key: str, boolean: bool) -> None:
        """
        Adds a bool criterion to the filter criteria.
        :param key: The key to filter by.
        :param boolean: The value of the criterion.
        """
        self.criteria[key] = boolean

    def add_range(self, key: str, min: int, max: int) -> None:
        """
        Adds a min-max criterion to the filter criteria.
        :param key: The key to filter by.
        :param min: The minimum value of the criterion.
        :param max: The maximum value of the criterion.
        """
        self.criteria[key] = (min, max)

    def custom_filter_key(self, item):
        """
        Returns True if the item fulfills the filter criteria.
        :param item: The item to check.
        """
        for key, value in self.criteria.items():
            if (
                isinstance(value, bool)
                and item[key] != value
                or isinstance(value, tuple)
                and not value[0] <= item[key] <= value[1]
            ):
                return False
        return True


def repos(
    amount: int = 100,
    sorting_criteria: RepositorySortCriteria = None,
    filter_criteria=None,
    keys_to_keep: list = None,
):
    """
    Returns repositories from GitHub that have a pom.xml file and fulfill the filter
    criteria. The returned repositories are sorted by the sorting criteria.
    """
    # Create equally weighted sorting criteria if it is not provided
    if sorting_criteria is None:
        sorting_criteria = RepositorySortCriteria()
        sorting_criteria.add("forks_count", 1)
        sorting_criteria.add("stargazers_count", 1)

    # Create filter criteria if it is not provided
    if filter_criteria is None:
        filter_criteria = RepositoryFilterCriteria()
        filter_criteria.add_bool("private", False)
        filter_criteria.add_bool("fork", False)
        filter_criteria.add_bool("archived", False)
        filter_criteria.add_bool("disabled", False)
        filter_criteria.add_range("stargazers_count", 100, MAX_INT)
        filter_criteria.add_range("forks_count", 100, MAX_INT)
        filter_criteria.add_range("watchers_count", 100, MAX_INT)
        filter_criteria.add_range("open_issues_count", 0, MAX_INT)
        filter_criteria.add_range("subscribers_count", 10, MAX_INT)

    # Get the repositories from GitHub
    per_page = 100
    if amount < 100:
        per_page = amount
    first_page = 1
    last_page = amount // per_page
    data = get_checkstyle_repos(per_page, first_page, last_page)

    # Filter out repositories based on criteria
    data = {
        key: value
        for key, value in data.items()
        if filter_criteria.custom_filter_key(value)
    }

    # Remove unnecessary keys if keys_to_keep is provided
    if keys_to_keep and len(keys_to_keep) > 0:
        all_keys = list(data.values())[0]
        keys_to_remove = [key for key in all_keys if key not in keys_to_keep]
        data = remove_keys(data, keys_to_remove)

    # Sort the repositories
    sorted_dict = sorted(data.values(), key=sorting_criteria.custom_sort_key)
    data = {item["full_name"]: item for item in sorted_dict}

    return data


def get_latest_main_commit(full_name: str, default_branch: str) -> str:
    """
    Returns the latest commit of the default branch of the repository.
    :param full_name: The full name of the repository.
    :param default_branch: The default branch of the repository.
    """
    # Get the latest commit of the default branch
    url = API_URL + "/" + REPOS + "/" + full_name + "/" + COMMITS + "/" + default_branch

    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        sha = response.json()["sha"]
    else:
        raise DownloadFailedException(
            f"Could not get the latest commit of the default branch of {full_name}."
        )
    return sha


def save_repos_as_json(data: dict[str, dict[str, str]], file_name: str) -> None:
    """
    Saves the given repos as a json file.
    :param data: The data to save.
    :param file_name: The name of the file.
    """
    with open(file_name, "w") as file:
        json.dump(data, file)


def save_repos_as_csv(data: dict[str, dict[str, str]], file_name: str) -> None:
    """
    Saves the given repos as a csv file.
    The csv file only contains the clone_url and the latest commit of the default
    branch.
    :param data: The data to save.
    :param file_name: The name of the file.
    """
    for repo in data.values():
        if "latest_commit" not in repo or repo["latest_commit"] is None:
            repo["latest_commit"] = get_latest_main_commit(
                repo["full_name"], repo["default_branch"]
            )

    with open(file_name, "w") as file:
        writer = csv.writer(file)
        for repo in data.values():
            writer.writerow([repo["clone_url"], repo["latest_commit"]])


def main():
    # Get the repositories
    data = repos(1)

    # Save the repositories
    save_repos_as_json(data, "repos.json")
    save_repos_as_csv(data, "repos.csv")


if __name__ == "__main__":
    main()
