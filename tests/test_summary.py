import pytest
import pandas as pd
import modin.pandas as modin

import data_describe as dd
from data_describe.core.summary import cardinality


@pytest.fixture
def load_summary(data):
    return dd.data_summary(data), data


@pytest.fixture
def load_modin_summary(modin_data):
    return mw.data_summary(modin_data), modin_data


@pytest.fixture
def load_modin_series_summary(modin_data):
    return mw.data_summary(modin_data.iloc[:,0], compute_backend='pandas')


def test_type(load_summary):
    summary = load_summary[0]
    assert isinstance(summary, pd.core.frame.DataFrame)


def test_modin_type(load_modin_summary):
    summary = load_modin_summary[0]
    assert isinstance(summary, modin.dataframe.DataFrame)


def test_modin_series(load_modin_series_summary):
    summary = load_modin_series_summary[0]
    assert isinstance(summary, pd.core.frame.DataFrame)


def test_shape(load_summary):
    summary, data = load_summary
    assert summary.shape == (9, data.shape[1])


def test_cardinality(data):
    assert cardinality(data.d) == 2
    assert cardinality(data.e) == 2


def test_pandas_series(data):
    assert dd.data_summary(data.iloc[:, 0]).shape == (9, 1)
