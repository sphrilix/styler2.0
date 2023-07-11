from src.styler2_0.download.repository import get_remaining_requests, repos


def test_repos():
    actual, last_downloaded_page = repos(amount=1)
    expected_key = "debezium/debezium"
    assert expected_key in actual


def test_get_remaining_requests():
    amount = get_remaining_requests()
    print(amount)
