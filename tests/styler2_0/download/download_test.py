import os
import shutil
import tempfile

from src.styler2_0.download.repository import (
    add_latest_commit,
    download_repos,
    filter_repos,
    get_remaining_requests,
    load_repos_from_json,
    save_repos_as_csv,
    save_repos_as_json,
)

CURR_DIR = os.path.dirname(os.path.relpath(__file__))
SAMPLE_PROJECT_5 = os.path.join(CURR_DIR, "../../res/sample_data")


def test_download_repos():
    actual, last_downloaded_page = download_repos(amount=1)
    expected_key = "debezium/debezium"
    assert expected_key in actual


def test_get_remaining_requests():
    amount = get_remaining_requests()
    assert amount > 0


def test_load_save_repos():
    save = tempfile.mkdtemp()

    filename = "repos_raw.json"
    data = load_repos_from_json(filename)
    save_repos_as_json(data, filename, dir_path=save)
    assert os.path.exists(os.path.join(save, filename))

    shutil.rmtree(save)


def test_add_latest_commit():
    filename = "repos_raw.json"
    data = load_repos_from_json(filename)
    data = add_latest_commit(data)
    assert "latest_commit" in data.get("debezium/debezium")


def test_filter_repos():
    filename = "repos_raw.json"
    data = load_repos_from_json(filename)
    data = filter_repos(data)
    assert len(data) == 1


def test_save_repos_as_csv():
    save = tempfile.mkdtemp()

    filename = "repos_raw.json"
    csv_filename = "repos.csv"

    data = load_repos_from_json(filename)
    data = add_latest_commit(data)
    data = filter_repos(data)
    save_repos_as_csv(data, csv_filename, dir_path=save)
    assert os.path.exists(os.path.join(save, csv_filename))

    shutil.rmtree(save)
