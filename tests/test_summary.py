import pytest

from data_describe.compat import _is_dataframe
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
    assert _is_dataframe(load_summary.summary)


@pytest.mark.base
def test_series(load_series_summary):
    assert isinstance(load_series_summary, SummaryWidget)
    assert _is_dataframe(load_series_summary.summary)


@pytest.mark.base
def test_shape(load_summary, compute_backend_df):
    assert load_summary.summary.shape == (compute_backend_df.shape[1], 10)
