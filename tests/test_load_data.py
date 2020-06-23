# import os
# from io import StringIO

# import pandas as pd
# from pandas.util.testing import assert_frame_equal
# import geopandas
# import tempfile
# import pytest

# import data_describe as mw
# from data_describe.utilities.load_data import download_gcs_file, read_file_type


# def test_local_dir():
#     data = mw.load_data("data/Addresses", encoding="latin1")
#     text_files = [file for file in os.listdir("data/Addresses") if ".txt" in file]
#     assert data.shape[0] == len(text_files)


# def test_local_csv():
#     data = mw.load_data("data/er_data.csv")
#     assert isinstance(data, pd.DataFrame)


# def test_local_json():
#     data = mw.load_data("data/Sarcasm_Headlines_Dataset.json")
#     assert isinstance(data, pd.DataFrame)


# def test_gcp(monkeypatch):
#     df1 = pd.DataFrame({"int": [1, 3]})

#     class MockGCSFileSystem:
#         def open(*args):
#             return StringIO(df1.to_csv(index=False))

#     monkeypatch.setattr("gcsfs.GCSFileSystem", MockGCSFileSystem)
#     df2 = mw.load_data(
#         filepath="gs://gcp-public-data-landsat/LC08/01/001/002/LC08_L1GT_001002_20160817_20170322_01_T2/LC08_L1GT_001002_20160817_20170322_01_T2_ANG.txt"
#     )
#     assert_frame_equal(df1, df2)


# def test_local_shapefile():
#     data = mw.load_data(filepath="data/geo/tl_2018_us_county.shp")
#     assert isinstance(data, geopandas.geodataframe.GeoDataFrame)


# def test_missing():
#     with pytest.raises(FileNotFoundError):
#         mw.load_data("this_does_not_exist.csv")


# def test_local_excel():
#     data = mw.load_data("data/Financial Sample.xlsx")
#     assert isinstance(data, pd.DataFrame)


# def test_gcs_file(monkeypatch):
#     class MockClient:
#         def __init__(self):
#             return

#         def bucket(self, bucket_name=None):
#             return MockBucket()

#     class MockBucket:
#         def __init__(self):
#             return

#         def list_blobs(self, prefix=None, max_results=None):
#             return [MockBlob()]

#     class MockBlob:
#         def __init__(self):
#             self.name = "data/geo/tl_2018_us_county.shp"

#         def download_to_filename(self, filename, client=None, start=None, end=None):
#             return

#     def Mockread_file(filepath):
#         return geopandas.geodataframe.GeoDataFrame()

#     def Mockgettempdir():
#         return "data/geo"

#     monkeypatch.setattr("google.cloud.storage.Client", MockClient)
#     monkeypatch.setattr(tempfile, "gettempdir", Mockgettempdir)
#     monkeypatch.setattr(geopandas, "read_file", Mockread_file)

#     shapefile_dir = download_gcs_file(filepath="gs://data/geo/tl_2018_us_county.shp")
#     assert isinstance(shapefile_dir, str)

#     geo_df = read_file_type(filepath="gs://data/geo/tl_2018_us_county.shp")
#     assert isinstance(geo_df, geopandas.geodataframe.GeoDataFrame)
