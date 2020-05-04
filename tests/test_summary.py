import pandas as pd
import pytest
import mwdata as mw
from mwdata.core.summary import cardinality


@pytest.fixture
def data_frame():
    df = pd.read_csv("data/er_data.csv")
    df.dropna(axis=1, inplace=True)
    return df


@pytest.fixture
def load_summary(data_frame):
    return mw.data_summary(data_frame)


def test_shape(load_summary):
    assert load_summary.shape == (9, 44)


def test_cardinality(data_frame):
    assert cardinality(data_frame.readmitted) == 2
    assert cardinality(data_frame.diag_1) == 458


def test_pandas_series(data_frame):
    assert mw.data_summary(data_frame.iloc[:, 0]).shape == (9, 1)

