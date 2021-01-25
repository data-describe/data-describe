import pytest
import numpy as np
import plotly
from pmdarima.arima import AutoARIMA

from data_describe.compat import _is_dataframe
from data_describe.anomaly.detection import (
    anomaly_detection,
    AnomalyDetectionWidget,
    _pandas_compute_anomaly,
    _pandas_compute_anomalies_stats,
    _stepwise_fit_and_predict,
)


@pytest.mark.base
def test_not_df():
    with pytest.raises(ValueError):
        anomaly_detection("this_is_a_string")


@pytest.mark.base
def test_method_not_implemented(numeric_data):
    with pytest.raises(ValueError):
        anomaly_detection(
            numeric_data,
            method="unimplemented",
        )


@pytest.mark.base
def test_regression_and_classification_not_implemented(numeric_data):
    with pytest.raises(ValueError):
        anomaly_detection(numeric_data, target="a", date_col=None)


@pytest.mark.base
def test_anomaly_widget():
    ad = AnomalyDetectionWidget()
    assert hasattr(ad, "method"), "Anomaly Detection Widget missing method"
    assert hasattr(ad, "estimator"), "Anomaly Detection Widget missing estimator"
    assert hasattr(ad, "time_split_index"), "Anomaly Detection Widget missing time_split_index"
    assert hasattr(ad, "viz_data"), "Anomaly Detection Widget missing viz_data"
    assert hasattr(ad, "ylabel"), "Anomaly Detection Widget missing ylabel"
    assert hasattr(ad, "xlabel"), "Anomaly Detection Widget missing xlabel"
    assert hasattr(ad, "target"), "Anomaly Detection Widget missing target"
    assert hasattr(ad, "date_col"), "Anomaly Detection Widget missing date_col"
    assert hasattr(ad, "n_periods"), "Anomaly Detection Widget missing n_periods"
    assert hasattr(ad, "sigma"), "Anomaly Detection Widget missing sigma"
    assert hasattr(ad, "__repr__"), "Anomaly Detection Widget missing __repr__ method"
    assert hasattr(
        ad, "_repr_html_"
    ), "Anomaly Detection Widget missing _repr_html_ method"
    assert hasattr(ad, "show"), "Anomaly Detection Widget missing show method"


auto_arima_args = {
    "start_p": 1,
    "start_q": 1,
    "max_p": 1,
    "max_q": 1,
    "m": 1,
    "start_P": 0,
    "seasonal": True,
    "d": 1,
    "D": 1,
    "trace": True,
    "error_action": "ignore",
    "suppress_warnings": True,
    "stepwise": True,
}


@pytest.fixture
def arima_default(numeric_data, **auto_arima_args):
    return anomaly_detection(
        numeric_data.iloc[:10],
        method="arima",
        target="a",
        date_col="index",
        time_split_index=5,
        n_periods=1,
        viz_backend="plotly",
        **auto_arima_args
    )


@pytest.mark.base
def test_arima_plotly(arima_default):
    figure = arima_default.show(viz_backend="plotly")
    assert isinstance(figure, plotly.graph_objs.Figure)


@pytest.fixture
def monkeypatch_auto_arima(monkeypatch):
    class mock_arima(AutoARIMA):
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.ones(X.shape[0])

    monkeypatch.setattr("pmdarima.arima.AutoARIMA", mock_arima)


@pytest.mark.base
def test_pandas_compute_anomaly(
    numeric_data, monkeypatch_auto_arima, **auto_arima_args
):
    widget = _pandas_compute_anomaly(
        numeric_data,
        method="arima",
        date_col="index",
        target="a",
        n_periods=1,
        **auto_arima_args
    )
    assert _is_dataframe(widget.viz_data)
    assert isinstance(
        widget, AnomalyDetectionWidget
    ), "Fit object was not a ClusterWidget"
    assert hasattr(widget, "estimator"), "Missing pmdarima auto_arima"
    assert hasattr(widget, "input_data"), "Missing input data"


@pytest.mark.base
def test_pandas_compute_anomalies_stats(numeric_data, arima_default):
    ad = arima_default
    assert isinstance(
        ad.show(viz_backend="plotly"), plotly.graph_objs.Figure
    ), "Default show() didn't return a plotly object"
    predictions_df = _stepwise_fit_and_predict(
        train=numeric_data["a"][:5],
        test=numeric_data["a"][5:10],
        n_periods=1,
        estimator=ad.estimator,
    )
    assert _is_dataframe(predictions_df)
    assert predictions_df.shape == (5, 2)

    predictions_df = _pandas_compute_anomalies_stats(
        predictions_df, n_periods=1, sigma=2
    )
    assert _is_dataframe(predictions_df)
    assert predictions_df.shape == (5, 17)
    assert len(predictions_df.columns) == len(ad.viz_data.columns)
