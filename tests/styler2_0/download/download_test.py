from src.styler2_0.download.repository import repos


def test_repos():
    actual = repos(amount=1)
    expected_key = "debezium/debezium"
    assert expected_key in actual
