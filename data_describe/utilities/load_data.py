import os
import tempfile

import pandas as pd

from data_describe.compat import _compat, requires
from data_describe.utilities.file_ext import _FileExtensionTypes, is_filetype


def load_data(filepath, all_folders=False, **kwargs):
    """Create pandas data frame from filepath.

    Args:
        filepath: The file path. Can be either a local filepath or Google Cloud Storage URI filepath
        all_folders: If True, searches for text files in nested folders. If False, looks for text files in the current folder
        kwargs: Keyword arguments to pass to the reader

            .csv, .json, and other: Uses pandas.read_csv or pandas.read_json

    Returns:
        A pandas data frame
    """
    if os.path.isfile(filepath):
        df = read_file_type(filepath, **kwargs)
    elif "gs://" in filepath:
        if _compat.check_install("gcsfs"):
            df = read_file_type(filepath, **kwargs)
        else:
            raise ImportError("Package gcsfs required to load from GCS")
    elif os.path.isdir(filepath):
        text = []
        encoding = kwargs.pop("encoding", "utf-8")
        if not all_folders:
            for file in os.listdir(filepath):
                if os.path.isfile(os.path.join(filepath, file)) and file.endswith(
                    ".txt"
                ):
                    with open(
                        os.path.join(filepath, file), "r", encoding=encoding
                    ) as f:
                        text.append(f.read())
        else:
            for root, _, files in os.walk(filepath):
                for file in files:
                    if file.endswith(".txt"):
                        with open(
                            os.path.join(root, file), "r", encoding=encoding
                        ) as f:
                            text.append(f.read())
        df = pd.DataFrame(text)
    else:
        raise FileNotFoundError("{} not a valid path".format(filepath))
    return df


def read_file_type(filepath, **kwargs):
    """Read the file based on file extension.

    Currently supports the following filetypes:
        csv, json, txt, shp
    Args:
        filepath: The filepath to open

        kwargs: Keyword arguments to pass to the reader
            .csv, .json, and other: Uses pandas.read_csv or pandas.read_json

    Returns:
        A Pandas data frame
    """
    extension = os.path.splitext(filepath)[1]
    if is_filetype(_FileExtensionTypes.CSV, extension):
        return pd.read_csv(filepath, **kwargs)
    elif is_filetype(_FileExtensionTypes.JSON, extension):
        lines = kwargs.pop("lines", True)
        return pd.read_json(filepath, lines=lines, **kwargs)
    elif is_filetype(_FileExtensionTypes.EXCEL, extension):
        return pd.read_excel(filepath, **kwargs)
    else:
        sep = kwargs.pop("sep", "\n")
        return pd.read_csv(filepath, sep=sep, **kwargs)


@requires("google.cloud.storage")
def download_gcs_file(filepath, bucket=None, prefix=None, **kwargs):
    """Downloads files from Google Cloud Storage.

    Args:
        filepath: The file path
        bucket: bucket to which the file belongs to in Google Cloud Storage
        prefix: parameter in list_blobs to limit the results to objects that have the specified prefix
        kwargs: Keyword arguments for list_blobs
    Returns:
        shapefile_dir: The shape file
    """
    client = _compat.storage.Client()
    bucket = client.bucket(bucket)
    max_results = kwargs.pop("max_results", None)
    blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
    tmpdir = tempfile.gettempdir()
    shapefile_dir = None
    for blob in blobs:
        fpath = blob.name
        fname = fpath.split("/")[-1]
        if len(fname) < 1:
            continue
        if fname in filepath:
            blob.download_to_filename(os.path.join(tmpdir, fname))
            if ".shp" in fname:
                shapefile_dir = os.path.join(tmpdir, fname)

    return shapefile_dir
