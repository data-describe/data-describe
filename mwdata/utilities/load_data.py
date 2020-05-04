import os
import pandas as pd


def load_data(filepath, kwargs=None):
    """ Create pandas data frame from filepath

    Args:
        filepath: The file path
        kwargs: Keyword arguments to pass to the reader
            .shp: Uses geopandas.read_file
            .csv, .json, and other: Uses pandas.read_csv or pandas.read_json

    Returns:
        A pandas data frame

    """
    text = []
    if os.path.isdir(filepath):
        encoding = kwargs.pop('encoding', 'utf-8')
        for file in os.listdir(filepath):
            with open(os.path.join(filepath, file), 'r', encoding=encoding) as f:
                text.append(f.read())
        df = pd.DataFrame(text)
    elif os.path.isfile(filepath):
        df = read_file_type(filepath, kwargs=kwargs)
    elif 'gs://' in filepath:
        df = read_file_type(filepath, kwargs=kwargs)
    else:
        raise FileNotFoundError('{} not a valid path'.format(filepath))
    return df


def read_file_type(filepath, kwargs=None):
    """ Read the file based on file extension

    Currently supports the following filetypes:
        csv, json, txt
    Args:
        filepath: The filepath to open
        kwargs: Keyword arguments to pass to the reader
            .shp: Uses geopandas.read_file
            .csv, .json, and other: Uses pandas.read_csv or pandas.read_json

    Returns:
        A Pandas data frame
    """
    extension = os.path.splitext(filepath)[1]
    if extension == '.csv':
        if kwargs is None:
            kwargs = {}
        return pd.read_csv(filepath, **kwargs)
    elif extension == '.json':
        if kwargs is None:
            kwargs = {'lines': True}
        return pd.read_json(filepath, **kwargs)
    elif extension == '.shp':
        import geopandas as gpd
        if kwargs is None:
            kwargs = {}
        return gpd.read_file(filepath, **kwargs)
    elif extension == '.xlsx':
        if kwargs is None:
            kwargs = {}
        return pd.read_excel(filepath, **kwargs)
    else:
        if kwargs is None:
            kwargs = {'sep': '\n'}
        return pd.read_csv(filepath, **kwargs)
