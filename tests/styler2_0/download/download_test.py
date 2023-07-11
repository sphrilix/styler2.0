from src.styler2_0.download.repository import download_repos, get_remaining_requests


def test_repos():
    actual, last_downloaded_page = download_repos(amount=1)
    expected_key = "debezium/debezium"
    assert expected_key in actual


def test_get_remaining_requests():
    amount = get_remaining_requests()
    print(amount)
