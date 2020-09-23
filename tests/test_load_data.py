import re
import os
from unittest.mock import patch, mock_open
import tempfile

import pandas as pd
import pytest

import data_describe as dd
from data_describe.misc.load_data import download_gcs_file, read_file_type


@pytest.fixture(autouse=True)
def skip_io(monkeypatch):
    def mock_read_csv(filepath_or_buffer, **kwargs):
        return pd.DataFrame()

    def mock_read_json(filepath_or_buffer, **kwargs):
        return pd.DataFrame()

    def mock_read_excel(filepath_or_buffer, **kwargs):
        return pd.DataFrame()

    def mock_isfile(path):
        return re.match(r".*\.[a-z]+", path)

    def mock_isdir(path):
        return not re.match(r".*\.[a-z]+", path)

    def mock_listdir(path):
        return ["file.txt"]

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)
    monkeypatch.setattr("pandas.read_json", mock_read_json)
    monkeypatch.setattr("pandas.read_excel", mock_read_excel)
    monkeypatch.setattr("os.path.isfile", mock_isfile)
    monkeypatch.setattr("os.path.isdir", mock_isdir)
    monkeypatch.setattr("os.listdir", mock_listdir)


@pytest.mark.base
def test_local_dir():
    with patch("builtins.open", mock_open(read_data="data")) as mock_file:
        dd.load_data("data/Addresses", encoding="latin1")
        mock_file.assert_called_with(
            os.path.join("data/Addresses", "file.txt"), "r", encoding="latin1"
        )


@pytest.mark.base
def test_local_csv():
    data = dd.load_data("data/er_data.csv")
    assert isinstance(data, pd.DataFrame)


@pytest.mark.base
def test_local_json():
    data = dd.load_data("data/Sarcasm_Headlines_Dataset.json")
    assert isinstance(data, pd.DataFrame)


def test_gcp(monkeypatch):
    df = dd.load_data(filepath="gs://file.txt")
    assert isinstance(df, pd.DataFrame)


@pytest.mark.base
def test_local_excel():
    data = dd.load_data("data/Financial Sample.xlsx")
    assert isinstance(data, pd.DataFrame)


def test_gcs_file(monkeypatch):
    class MockClient:
        def __init__(self):
            return

        def bucket(self, bucket_name=None):
            return MockBucket()

    class MockBucket:
        def __init__(self):
            return

        def list_blobs(self, prefix=None, max_results=None):
            return [MockBlob()]

    class MockBlob:
        def __init__(self):
            self.name = "data/geo/tl_2018_us_county.shp"

        def download_to_filename(self, filename, client=None, start=None, end=None):
            return

    def Mockread_file(filepath):
        return pd.GeoDataFrame()

    def Mockgettempdir():
        return "data/geo"

    monkeypatch.setattr("google.cloud.storage.Client", MockClient)
    monkeypatch.setattr(tempfile, "gettempdir", Mockgettempdir)

    file_dir = download_gcs_file(filepath="gs://data/geo/tl_2018_us_county.shp")
    assert isinstance(file_dir, str)

    df = read_file_type(filepath="gs://file.csv")
    assert isinstance(df, pd.DataFrame)
