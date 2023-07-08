import requests

API_URL = "https://api.github.com/search/"
CODE = "code"
REPOSITORIES = "repositories"
AUTH_TOKEN = "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"


class DownloadFailedException(Exception):
    """
    Exception that is thrown whenever the download failed.
    """


def get_checkstyle_repos(
    per_page: int = 100, first_page: int = 1, last_page: int = 1
) -> dict[str, dict[str, str]]:
    # Define the API endpoint and query parameters
    api_url = API_URL + CODE
    query_params = {
        "q": "checkstyle filename:pom.xml",
        "per_page": per_page,
        "page": first_page,
    }

    # Set the Authorization header with the token
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

    data = {}

    while True:
        # Make the GET request
        response = requests.get(api_url, params=query_params, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            # Convert the response json to a python dictionary and add it to the data
            search_response_json = response.json()
            for item in search_response_json["items"]:
                repository = item["repository"]
                repository_name = repository["full_name"]
                repository_url = repository["url"]

                # Open the repository url and save the json
                repository_response = requests.get(repository_url, headers=headers)
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
    The sort criteria is a list of triples (key, ascending, weight).
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


def repos(amount: int = 100, sorting_criteria: RepositorySortCriteria = None):
    """
    Returns repositories from GitHub that fulfill the following criteria:
    - The repository contains a pom.xml file.
    - The repository contains a checkstyle configuration file.
    The returned repositories are sorted by the number of forks and stars.
    """
    # Create equally weighted sorting criteria if it is not provided
    if sorting_criteria is None:
        sorting_criteria = RepositorySortCriteria()
        sorting_criteria.add("forks_count", 1)
        sorting_criteria.add("stargazers_count", 1)

    # Get the repositories from GitHub
    per_page = 100
    if amount < 100:
        per_page = amount
    first_page = 1
    last_page = amount // per_page
    data = get_checkstyle_repos(per_page, first_page, last_page)

    # Remove unnecessary keys
    all_keys = list(data.values())[0]
    keys_to_keep = ["full_name", "html_url", "stargazers_count", "forks_count"]
    keys_to_remove = [key for key in all_keys if key not in keys_to_keep]
    data = remove_keys(data, keys_to_remove)

    # Sort the repositories
    sorted_dict = sorted(data.values(), key=sorting_criteria.custom_sort_key)
    data = {item["full_name"]: item for item in sorted_dict}

    return data
