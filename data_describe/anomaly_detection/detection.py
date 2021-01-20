from typing import Optional  # , Union
import logging

# import seaborn as sns
# from matplotlib.figure import Figure
# from plotly.subplots import make_subplots
# from plotly.offline import init_notebook_mode

import numpy as np
import plotly.graph_objs as go
import plotly.offline as po
import pandas as pd

from data_describe.backends import _get_compute_backend, _get_viz_backend
from data_describe.compat import _is_dataframe, _requires, _in_notebook
from data_describe._widget import BaseWidget

# from data_describe.config._config import get_option


class AnomalyDetectionWidget(BaseWidget):
    """Interface for collecting additional information about the anomaly detection widget."""

    def __init__(
        self,
        estimator=None,
        method=None,
        viz_data=None,
        split_data=None,
        **kwargs,
    ):
        super(AnomalyDetectionWidget, self).__init__(**kwargs)
        self.estimator = estimator
        self.method = method
        self.input_data = None
        self.viz_data = viz_data
        self.xlabel = None
        self.ylabel = None
        self.split_data = split_data

    def __str__(self):
        return "data-describe Anomaly Detection Widget"

    def __repr__(self):
        return f"Anomaly Widget using {self.method}"

    def show(self, viz_backend=None, **kwargs):
        """The default display for this output.

        Displays the anomalies, projected as a lineplot, with detected anomalies as red markers

        Args:
            viz_backend: The visualization backend.
            **kwargs: Keyword arguments.

        Raises:
            ValueError: Data to visualize is missing / not calculated.

        Returns:
            The anomaly plot.
        """
        backend = viz_backend or self.viz_backend

        if self.viz_data is None:
            raise ValueError("Could not find data to visualize.")

        # TODO(truongc2): update labels
        # return _get_viz_backend(backend).viz_plot_anomaly(
        #     self.viz_data, xlabel=self.xlabel, ylabel=self.ylabel, **kwargs,
        # )
        return _get_viz_backend(backend).viz_plot_anomaly(self.viz_data, **kwargs)


def anomaly_detection(
    data,
    target: str = None,
    ml_use_case: str = "timeseries",
    date_col: str = None,
    method="arima",
    estimator=None,
    compute_backend: Optional[str] = None,
    viz_backend: Optional[str] = None,
    periods=None,
    data_split=50,
    **kwargs,
):
    """Identify and mark anamolies.

    Args:
        data (DataFrame): The dataframe
        target (str): The target column
        ml_use_case (str): Choose between classification, regression, or timeseries.
        date_col (str): Target column or datetime column
        method (str, optional): Select method from this list. Select from [arima, WER, iforest]
        estimator: Fitted estimator. Must have a .predict() function. Defaults to None.
        periods (int): Number of periods.
        data_split (int): Index to split data into train and test.
        compute_backend (str): Select compute backend.
        viz_backend (str): Select compute backend.
        **kwargs: Keyword arguments.

    Return:
        AnomalyWidget
    """
    # checks if input is dataframe
    if not _is_dataframe(data):
        raise ValueError("Data frame required")

    # checks if estimator exists and if it follows the general sklearn guidelines
    if estimator:
        if not hasattr(estimator, "predict") and not hasattr(estimator, "fit"):
            raise AttributeError(
                "Input model does not contain the 'predict' or 'fit' method."
            )

    ml_methods = {
        "timeseries": ["arima"],
    }

    if ml_use_case.lower() not in ml_methods.keys():
        return ValueError(
            "Please choose from classification, regression, or timeseries"
        )

    if method.lower() not in ml_methods[ml_use_case]:
        raise ValueError(
            f"{method} not implemented Please choose one of the following: {ml_methods[ml_use_case]}"
        )

    # ensures date_col is a datetime object and sets as datetimeindex
    if date_col:
        numeric_data = data.select_dtypes("number")
        numeric_data.index = pd.to_datetime(data[date_col], unit="ns")

        # supervised
        # if target:

        # # unsupervised
        # else:
        # arguments to differ :
        #     target, date_col, estimator, method, periods
        anomalywidget = _get_compute_backend(
            compute_backend, numeric_data
        ).compute_anomaly(
            data=numeric_data,
            target=target,
            date_col=date_col,
            estimator=estimator,
            method=method,
            periods=periods,
            data_split=data_split,
            # ml_use_case=ml_use_case,
            **kwargs,
        )

    # anomalywidget = _get_viz_backend(viz_backend).viz_plot_anomaly(
    #     anomalywidget.viz_data, **kwargs
    # )

    anomalywidget.viz_backend = viz_backend

    return anomalywidget


def _pandas_compute_anomaly(
    data, date_col, target, method, data_split, periods=None, estimator=None, **kwargs
):
    """Backend implementation of cluster.

    Args:
        data (DataFrame): The data
        method (str): The algorithm
        **kwargs: Keyword arguments.

    Raises:
        ValueError: If method is not implemented

    Returns:
        (clusters, ClusterFit)

        clusters: The predicted cluster labels
        ClusterFit: A class containing additional information about the fit
    """
    # arguments to differ :
    #     target, date_col, estimator, method, periods
    if date_col:  # timeseries
        if target:  # supervised
            train, test = (
                data[target][0:data_split],
                data[target][data_split:],
            )
            if not estimator:
                from pmdarima.arima import auto_arima

                logging.info("Estimator was not specified. Defaulting to ARIMA.")
                estimator = auto_arima(
                    train,
                    start_p=1,
                    start_q=1,
                    max_p=3,
                    max_q=3,
                    m=7,
                    start_P=0,
                    seasonal=True,
                    d=1,
                    D=1,
                    trace=True,
                    error_action="ignore",
                    suppress_warnings=True,
                    stepwise=True,
                )

            # make one-step forecast
            predictions_df = ts_predictor(
                train=train, test=test, periods=periods, estimator=estimator
            )

            predictions_df = _pandas_compute_anomalies_stats(
                predictions_df, date_col, window=periods
            )

        else:  # unsupervised
            raise ValueError("Work in progress for unsupervised methods")

    else:  # not timeseries
        raise ValueError("Work in progress for non timeseries data")

    # elif method == "wer":
    #     raise ValueError("Work in progress")

    return AnomalyDetectionWidget(
        estimator=estimator,
        method=method,
        viz_data=predictions_df,
        data_split=data_split,
    )


def ts_predictor(train, test, periods, estimator):
    """Perform stepwise fit and predict.

    Args:
        train ([type]): [description]
        test ([type]): [description]
        periods ([type]): [description]
        estimator ([type]): [description]

    Returns:
        [type]: [description]
    """
    history = [x for x in train]
    predictions = list()
    for t in test.index:
        estimator.fit(history)
        output = estimator.predict(n_periods=periods)
        predictions.append(output[0])
        obs = test[t]
        history.append(obs)

    predictions_df = pd.DataFrame()
    predictions_df["actuals"] = test  # test[target]
    predictions_df["predictions"] = predictions
    return predictions_df


def _pandas_compute_anomalies_stats(predictions_df, date_col, window=7):
    if (
        "actuals" not in predictions_df.columns
        and "predictions" not in predictions_df.columns
    ):
        raise ValueError("'actuals' and 'predictions' are not found in the dataframe")
    predictions_df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    predictions_df.fillna(0, inplace=True)
    predictions_df["error"] = predictions_df["actuals"] - predictions_df["predictions"]
    predictions_df["percentage_change"] = (
        predictions_df["error"] / predictions_df["actuals"]
    ) * 100
    predictions_df["meanval"] = predictions_df["error"].rolling(window=window).mean()
    predictions_df["deviation"] = predictions_df["error"].rolling(window=window).std()
    predictions_df["-3s"] = predictions_df["meanval"] - (
        2 * predictions_df["deviation"]
    )
    predictions_df["3s"] = predictions_df["meanval"] + (2 * predictions_df["deviation"])
    predictions_df["-2s"] = predictions_df["meanval"] - (
        1.75 * predictions_df["deviation"]
    )
    predictions_df["2s"] = predictions_df["meanval"] + (
        1.75 * predictions_df["deviation"]
    )
    predictions_df["-1s"] = predictions_df["meanval"] - (
        1.5 * predictions_df["deviation"]
    )
    predictions_df["1s"] = predictions_df["meanval"] + (
        1.5 * predictions_df["deviation"]
    )
    cut_list = predictions_df[
        ["error", "-3s", "-2s", "-1s", "meanval", "1s", "2s", "3s"]
    ]

    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    predictions_df["impact"] = [
        (lambda x: np.where(cut_sort == predictions_df["error"][x])[1][0])(
            x
        )  # update x+50
        for x in range(len(predictions_df["error"]))
    ]
    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
    region = {
        0: "NEGATIVE",
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEGATIVE",
        4: "POSITIVE",
        5: "POSITIVE",
        6: "POSITIVE",
        7: "POSITIVE",
    }
    predictions_df["color"] = predictions_df["impact"].map(severity)
    predictions_df["region"] = predictions_df["impact"].map(region)
    predictions_df["anomaly_points"] = np.where(
        predictions_df["color"] == 3, predictions_df["error"], np.nan
    )
    predictions_df = predictions_df.sort_index(ascending=False)

    return predictions_df


@_requires("plotly")
def _plotly_viz_anomaly(predictions_df, marker_color="red"):
    # fix scales, actuals do not meet with anomalies
    predictions_df = predictions_df.iloc[:-6, :]  # update the -6
    # predictions_df.reset_index(inplace=True)
    bool_array = abs(predictions_df["anomaly_points"]) > 0
    actuals = predictions_df["actuals"][-len(bool_array) :]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan

    # color_map = {0: "'rgba(228, 222, 249, 0.65)'", 1: "yellow", 2: "orange", 3: "red"}

    anomalies = go.Scatter(
        name="Anomaly",
        x=predictions_df.index,
        xaxis="x1",
        yaxis="y1",
        y=predictions_df["anomaly_points"],
        mode="markers",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )

    upper_bound = go.Scatter(
        hoverinfo="skip",
        x=predictions_df.index,
        showlegend=False,
        xaxis="x1",
        yaxis="y1",
        y=predictions_df["3s"],
        marker=dict(color="#444"),
        line=dict(color=("rgb(23, 96, 167)"), width=2, dash="dash"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    lower_bound = go.Scatter(
        name="Confidence Interval",
        x=predictions_df.index,
        xaxis="x1",
        yaxis="y1",
        y=predictions_df["-3s"],
        marker=dict(color="#444"),
        line=dict(color=("rgb(23, 96, 167)"), width=2, dash="dash"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    Actuals = go.Scatter(
        name="Actuals",
        x=predictions_df.index,
        y=predictions_df["actuals"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="blue"),
    )

    Predicted = go.Scatter(
        name="Predicted",
        x=predictions_df.index,
        y=predictions_df["predictions"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="orange"),
    )

    # create plot for error...
    Error = go.Scatter(
        name="Error",
        x=predictions_df.index,
        y=predictions_df["error"],
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="red"),
        text="Error",
    )

    anomalies_map = go.Scatter(
        name="anomaly actual",
        showlegend=False,
        x=predictions_df.index,
        y=anomaly_points,
        mode="markers",
        xaxis="x2",
        yaxis="y2",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )

    Mvingavrg = go.Scatter(
        name="Moving Average",
        x=predictions_df.index,
        y=predictions_df["meanval"],
        # mode='line',
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="green"),
        text="Moving average",
    )

    axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4,
        gridcolor="#ffffff",
        tickfont=dict(size=10),
    )

    layout = dict(
        width=1000,
        height=865,
        autosize=False,
        title="ARIMA Anomalies",
        margin=dict(t=75),
        showlegend=True,
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor="y1", showticklabels=True)),
        xaxis2=dict(axis, **dict(domain=[0, 1], anchor="y2", showticklabels=True)),
        yaxis1=dict(
            axis,
            **dict(domain=[2 * 0.21 + 0.20 + 0.09, 1], anchor="x1", hoverformat=".2f"),
        ),
        yaxis2=dict(
            axis,
            **dict(
                domain=[0.21 + 0.12, 2 * 0.31 + 0.02], anchor="x2", hoverformat=".2f"
            ),
        ),
    )

    fig = go.Figure(
        data=[
            anomalies,
            anomalies_map,  # table
            upper_bound,
            lower_bound,
            Actuals,
            Predicted,
            Mvingavrg,
            Error,
        ],
        layout=layout,
    )

    if _in_notebook():
        po.init_notebook_mode(connected=True)
        return po.iplot(fig)
    else:
        return fig
