from typing import Optional
import logging

import numpy as np
import plotly.graph_objs as go
import plotly.offline as po
import pandas as pd

from data_describe.backends import _get_compute_backend, _get_viz_backend
from data_describe.compat import _is_dataframe, _requires, _in_notebook, _compat
from data_describe._widget import BaseWidget


class AnomalyDetectionWidget(BaseWidget):
    """Container for anomaly calculations and visualization.

    This class (object) is returned from the ``anomaly_detection`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        method (str, optional): {'arima'} The type of anomaly detection algorithm.
        estimator: The anomaly detection estimator/model.
        time_split_index (str, optional): The index to split the input data into a training set and testing set.
            Note: Data is not shuffled.
        viz_data (DataFrame): The data used for the default visualization.
        input_data (DataFrame): The input data.
        xlabel (str): The x-axis label for the anomaly plot.
        ylabel (str): The y-axis label for the anomaly plot.
        target (str, optional): The target column.
        date_col (str, optional): The date column.
        n_periods (int, optional): The number of periods for timeseries window.
    """

    def __init__(
        self,
        estimator=None,
        method=None,
        viz_data=None,
        time_split_index=None,
        n_periods=None,
        **kwargs,
    ):
        """Anomaly Detection.

        Args:
            method (str, optional): {'arima'} The type of anomaly detection algorithm.
            estimator: The anomaly detection estimator/model.
            time_split_index (str, optional): The index to split the input data into a training set and testing set.
                Note: Data is not shuffled.
            viz_data (DataFrame): The data used for the default visualization.
            input_data (DataFrame): The input data.
            xlabel (str): The x-axis label for the anomaly plot.
            ylabel (str): The y-axis label for the anomaly plot.
            target (str, optional): The target column.
            date_col (str, optional): The date column.
            n_periods (int, optional): The number of periods for timeseries window.
            **kwargs: Keyword arguments.
        """
        super(AnomalyDetectionWidget, self).__init__(**kwargs)
        self.method = method
        self.estimator = estimator
        self.time_split_index = time_split_index
        self.viz_data = viz_data
        self.n_periods = n_periods
        self.input_data = None
        self.xlabel = None
        self.ylabel = None
        self.target = None
        self.date_col = None
        self.sigma = None

    def __str__(self):
        return "data-describe Anomaly Detection Widget"

    def __repr__(self):
        return f"Anomaly Widget using {self.method}"

    def show(self, viz_backend=None, **kwargs):
        """The default display for this output.

        Displays the anomalies, projected as a lineplot or scatterplot, with detected anomalies as red markers

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

        return _get_viz_backend(backend).viz_plot_anomaly(
            predictions_df=self.viz_data,
            n_periods=self.n_periods,
            xlabel=self.date_col,
            ylabel=self.target,
            **kwargs,
        )


def anomaly_detection(
    data,
    target: Optional[str] = None,
    date_col: Optional[str] = None,
    method: str = "arima",
    estimator=None,
    time_split_index: Optional[int] = None,
    n_periods: Optional[int] = None,
    sigma: float = 2.0,
    compute_backend: Optional[str] = None,
    viz_backend: Optional[str] = None,
    **kwargs,
):
    """Identify and mark anamolies.

    This feature identifies anomalies in timeseries and tabular data using multiple approaches (supervised, unsupervised, and statistical)
    and then projects the data onto a plot for visualization.

    Args:
        data (DataFrame): The dataframe.
        target (str, optional): The target column. Defaults to None. If target is None, unsupervised methods are used.
        date_col (str, optional): The datetime column. If date_col is specified, the data will be treated as timeseries.
            If the data does not contain datetimes, but contains sequences, set date_col = 'index'.
        method (str, optional): Select method from the list: ["arima"]. Only "arima" is supported.
        estimator (optional): Fitted or instantiated estimator with a .predict() and .fit() method. Defaults to None.
            If estimator is instantiated but not fitted, time_split_index must be specified.
        time_split_index (int, optional): Index to split the data into a train set and a test set. Defaults to None.
            time_split_index must be specified if estimator is instantiated but not fitted.
        n_periods (int, optional): Number of periods for timeseries anomaly detection. Defaults to None.
        sigma (float, optional): The standard deviation requirement for identifying anomalies. Defaults to 2.
        compute_backend (str, optional): The compute backend.
        viz_backend (str, optional): The visualization backend.
        **kwargs: Keyword arguments.

    Return:
        AnomalyWidget
    """
    # checks if input is dataframe
    if not _is_dataframe(data):
        raise ValueError("Data frame required")

    # checks if estimator exists and if it follows the .predict and .fit methods
    if estimator:
        if not hasattr(estimator, "predict") and not hasattr(estimator, "fit"):
            raise AttributeError(
                "Input model does not contain the 'predict' or 'fit' method."
            )
    # TODO(truongc2): Update available methods
    # ml_methods = {
    #     "timeseries": ["arima"],
    # }
    anomalywidget = _get_compute_backend(compute_backend, data).compute_anomaly(
        data=data,
        target=target,
        date_col=date_col,
        estimator=estimator,
        n_periods=n_periods,
        time_split_index=time_split_index,
        method=method,
        sigma=sigma,
        **kwargs,
    )

    return anomalywidget


def _pandas_compute_anomaly(
    data,
    target: Optional[str] = None,
    date_col: Optional[str] = None,
    method: Optional[str] = "arima",
    estimator=None,
    n_periods: Optional[int] = None,
    time_split_index: Optional[int] = None,
    **kwargs,
):
    """Backend implementation of anomaly detection.

    Args:
        data (DataFrame): The dataframe.
        target (str, optional): The target column. Defaults to None. If target is None, unsupervised methods are used.
        date_col (str, optional): Datetime column if data is timeseries. Defaults to None. If data is timeseries, date_col must be specified.
        method (str, optional): Select method from this list. Only "arima" is supported.
        estimator (optional): Fitted or instantiated estimator with a .predict() and .fit() method. Defaults to None.
            If estimator is instantiated but not fitted, time_split_index must be specified.
        n_periods (int, optional): Number of periods for timeseries anomaly detection. Default is None.
        time_split_index (int, optional): Index to split the data into a train set and a test set. Defaults to None.
            time_split_index must be specified if estimator is instantiated but not fitted.
        **kwargs: Keyword arguments.

    Raises:
        ValueError: If method is not implemented.

    Returns:
        AnomalyDetectionWidget

    """
    # Timeseries indicator
    # ensures date_col is a datetime object and sets as datetimeindex
    if not date_col:
        raise ValueError("Please specify data_col for timeseries anomaly detection.")

    if date_col != "index":
        data.index = pd.to_datetime(data[date_col])

    numeric_data = data.select_dtypes("number")

    if not numeric_data.index.is_monotonic_increasing:
        numeric_data.sort_index(inplace=True)

    # Supervised learning indicator
    if not target:
        raise ValueError(
            "Unsupervised timeseries methods for anomaly detection are not yet supported. Please specify the target argument to continue."
        )
    train, test = (
        numeric_data[target][0:time_split_index],
        numeric_data[target][time_split_index:],
    )

    # Default to ARIMA model
    if not estimator:
        estimator = _compat["pmdarima.arima"].auto_arima(
            train, random_state=1, **kwargs
        )

    # make one-step forecast
    predictions_df = _stepwise_fit_and_predict(
        train=train, test=test, estimator=estimator
    )

    # Post-processing for errors and confidence interval
    predictions_df = _pandas_compute_anomalies_stats(predictions_df, window=n_periods)

    return AnomalyDetectionWidget(
        estimator=estimator,
        method=method,
        viz_data=predictions_df,
        time_split_index=time_split_index,
        target=target,
        n_periods=n_periods,
    )


def _stepwise_fit_and_predict(train, test, estimator):
    """Perform stepwise fit and predict for timeseries data.

    Args:
        train (Series): The training data.
        test (Series): The testing data.
        estimator: The estimator.

    Returns:
        predictions_df: DataFrame containing the ground truth, predictions, and indexed by the datetime.
    """
    logging.info(
        "Performing stepwise fit and prediction using ARIMA. This may take a couple minutes..."
    )
    history = train.to_list()
    predictions = list()
    for t in test.index:
        estimator.fit(history)
        output = estimator.predict(n_periods=1)
        predictions.append(output[0])
        obs = test[t]
        history.append(obs)

    predictions_df = pd.DataFrame()
    predictions_df["actuals"] = test
    predictions_df["predictions"] = predictions
    return predictions_df


def _pandas_compute_anomalies_stats(predictions_df, window, sigma=2):
    """Detects anomalies based on the statistical profiling of the residuals (actuals - predicted).

    The rolling mean and rolling standard deviation is used to identify points that are more than 2 standard deviations away from the mean.

    Args:
        predictions_df (DataFrame): The dataframe containing the ground truth and predictions.
        window (int, optional): Size of the moving window. This is the number of observations used for calculating the statistic.
        sigma (float). The standard deviation requirement for anomalies.

    Raises:
        ValueError: If 'actuals' and 'predictions' are not found in predictions_df.

    Returns:
        predictions_df: The dataframe containing the predictions and computed statistics.

    """
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
        sigma * predictions_df["deviation"]
    )
    predictions_df["3s"] = predictions_df["meanval"] + (
        sigma * predictions_df["deviation"]
    )
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

    # TODO(truongc2): Find a more robust way to call the index
    if not isinstance(predictions_df.index, pd.core.indexes.datetimes.DatetimeIndex):
        predictions_df.reset_index(inplace=True)

    predictions_df["impact"] = [
        (lambda x: np.where(cut_sort == predictions_df["error"][x])[1][0])(x)
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
def _plotly_viz_anomaly(
    predictions_df,
    n_periods,
    ylabel,
    xlabel="Time",
    marker_color="red",
):
    """Visualize anomalies using plotly.

    Args:
        predictions_df (DataFrame): The dataframe containing the ground truth, predictions, and statistics.
        n_periods (int):
        marker_color (str): The color to mark anomalies. Defaults to "red".

    Returns:
        Plotly plot

    """
    lookback = -1 * (n_periods - 1)
    predictions_df = predictions_df.iloc[:lookback, :]
    bool_array = abs(predictions_df["anomaly_points"]) > 0
    actuals = predictions_df["actuals"][-len(bool_array) :]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan

    anomalies = go.Scatter(
        name="Anomaly",
        x=predictions_df.index,
        y=predictions_df["anomaly_points"],
        xaxis="x1",
        yaxis="y1",
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

    actuals = go.Scatter(
        name="Actuals",
        x=predictions_df.index,
        y=predictions_df["actuals"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="blue"),
    )

    predicted = go.Scatter(
        name="Predicted",
        x=predictions_df.index,
        y=predictions_df["predictions"],
        xaxis="x2",
        yaxis="y2",
        marker=dict(size=12, line=dict(width=1), color="orange"),
    )

    # create plot for error...
    error = go.Scatter(
        name="Error",
        x=predictions_df.index,
        y=predictions_df["error"],
        xaxis="x1",
        yaxis="y1",
        marker=dict(size=12, line=dict(width=1), color="red"),
        text="Error",
    )

    anomalies_map = go.Scatter(
        name="Anomaly",
        showlegend=False,
        x=predictions_df.index,
        y=anomaly_points,
        mode="markers",
        xaxis="x2",
        yaxis="y2",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )

    moving_average = go.Scatter(
        name="Moving Average",
        x=predictions_df.index,
        y=predictions_df["meanval"],
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
        # title="ARIMA Anomalies",
        yaxis_title="ylabel",
        xaxis_title="xlabel",
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
            anomalies_map,
            upper_bound,
            lower_bound,
            actuals,
            predicted,
            moving_average,
            error,
        ],
        layout=layout,
    )
    fig.update_layout(
        title="ARIMA Anomalies",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title="Legend",
    )
    if _in_notebook():
        po.init_notebook_mode(connected=True)
        return po.iplot(fig)
    else:
        return fig
