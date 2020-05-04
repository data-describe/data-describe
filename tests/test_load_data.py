import pandas as pd
from pandas.util.testing import assert_frame_equal
import mwdata as mw
import os
from io import StringIO
import geopandas
import pytest


def test_local_dir():
    data = mw.load_data('data/Addresses', kwargs={'encoding': 'latin1'})
    assert data.shape[0] == len(os.listdir('data/Addresses'))


def test_local_csv():
    data = mw.load_data('data/er_data.csv')
    assert isinstance(data, pd.DataFrame)


def test_local_json():
    data = mw.load_data('data/Sarcasm_Headlines_Dataset.json')
    assert isinstance(data, pd.DataFrame)


def test_gcp(monkeypatch):
    df1 = pd.DataFrame({'int': [1, 3]})

    class MockGCSFileSystem:
        def open(*args):
            return StringIO(df1.to_csv(index=False))

    monkeypatch.setattr('gcsfs.GCSFileSystem', MockGCSFileSystem)
    df2 = mw.load_data(filepath='gs://gcp-public-data-landsat/LC08/01/001/002/LC08_L1GT_001002_20160817_20170322_01_T2/LC08_L1GT_001002_20160817_20170322_01_T2_ANG.txt')
    assert_frame_equal(df1, df2)


def test_local_shapefile():
    data = mw.load_data(filepath='data/geo/tl_2018_us_county.shp')
    assert isinstance(data, geopandas.geodataframe.GeoDataFrame)


def test_missing():
    with pytest.raises(FileNotFoundError):
        mw.load_data('this_does_not_exist.csv')


def test_local_excel():
    data = mw.load_data('data/Financial Sample.xlsx')
    assert isinstance(data, pd.DataFrame)
