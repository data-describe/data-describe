import pytest

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.core.summary import data_summary, SummaryWidget


@pytest.fixture
def load_summary(compute_backend_df):
    return data_summary(compute_backend_df)


@pytest.fixture
def load_series_summary(compute_backend_df):
    return data_summary(compute_backend_df.iloc[:, 0])


@pytest.mark.base
def test_type(load_summary):
    assert isinstance(load_summary, SummaryWidget)
    assert isinstance(load_summary.summary, _DATAFRAME_TYPE)


@pytest.mark.base
def test_series(load_series_summary):
    assert isinstance(load_series_summary, SummaryWidget)
    assert isinstance(load_series_summary.summary, _DATAFRAME_TYPE)


@pytest.mark.base
def test_shape(load_summary, compute_backend_df):
    assert load_summary.summary.shape == (9, compute_backend_df.shape[1])
