import pytest

from data_describe.compat import _DATAFRAME_TYPE
from data_describe.core.summary import data_summary


@pytest.fixture
def load_summary(compute_backend_df):
    return data_summary(compute_backend_df), compute_backend_df


@pytest.fixture
def load_series_summary(compute_backend_df):
    return (
        data_summary(compute_backend_df.iloc[:, 0]),
        compute_backend_df.iloc[:, 0],
    )


@pytest.mark.base
def test_type(load_summary):
    summary = load_summary[0]
    assert isinstance(summary, _DATAFRAME_TYPE)


@pytest.mark.base
def test_series(load_series_summary):
    summary = load_series_summary[0]
    assert isinstance(summary, _DATAFRAME_TYPE)


@pytest.mark.base
def test_shape(load_summary):
    summary, data = load_summary
    assert summary.shape == (9, data.shape[1])
