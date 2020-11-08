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
@pytest.mark.parametrize(
    "compute_backend_df",
    ["pandas", pytest.param("modin.pandas", marks=pytest.mark.xfail)],
    indirect=True,
)  # xfail: modin #2376
def test_dataframe_attributes(load_summary, compute_backend_df):
    assert isinstance(load_summary, SummaryWidget)
    assert _is_dataframe(load_summary.input_data)
    assert _is_dataframe(load_summary.info_data)
    assert _is_dataframe(load_summary.summary_data)
    assert hasattr(load_summary, "as_percentage")
    assert hasattr(load_summary, "auto_float")


@pytest.mark.base
@pytest.mark.parametrize(
    "compute_backend_df",
    ["pandas", pytest.param("modin.pandas", marks=pytest.mark.xfail)],
    indirect=True,
)  # xfail: modin #2376
def test_series_attributes(load_series_summary, compute_backend_df):
    assert isinstance(load_series_summary, SummaryWidget)
    assert _is_dataframe(load_series_summary.input_data)
    assert _is_dataframe(load_series_summary.info_data)
    assert _is_dataframe(load_series_summary.summary_data)
    assert hasattr(load_series_summary, "as_percentage")
    assert hasattr(load_series_summary, "auto_float")


@pytest.mark.base
@pytest.mark.parametrize(
    "compute_backend_df",
    ["pandas", pytest.param("modin.pandas", marks=pytest.mark.xfail)],
    indirect=True,
)  # xfail: modin #2376
def test_shape(load_summary, compute_backend_df):
    assert load_summary.summary_data.shape == (compute_backend_df.shape[1], 10)
    assert load_summary.info_data.shape == (3, 1)
