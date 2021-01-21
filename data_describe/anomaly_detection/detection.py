from typing import Optional  # , Union

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
    """Container for anomaly calculations and visualization.

    This class (object) is returned from the ``anomaly_detection`` function. The
    attributes documented below can be accessed or extracted.

    Attributes:
        method (str, optional): {'arima'} The type of anomaly detection algorithm.
        estimator: The anomaly detection estimator/model.
        data_split (str, optional): The index to split the input data into a training set and testing set.
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
        data_split=None,
        **kwargs,
    ):
        """Anomaly Detection.

        Args:
            method (str, optional): {'arima'} The type of anomaly detection algorithm.
            estimator: The anomaly detection estimator/model.
            data_split (str, optional): The index to split the input data into a training set and testing set.
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
        self.data_split = data_split
        self.viz_data = viz_data
        self.input_data = None
        self.xlabel = None
        self.ylabel = None
        self.target = None
        self.date_col = None
        self.n_periods = None
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
    data_split: Optional[int] = None,
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
        method (str, optional): Select method from this list. Only "arima" is supported.
        estimator (optional): Fitted or instantiated estimator with a .predict() and .fit() method. Defaults to None.
            If estimator is instantiated but not fitted, data_split must be specified.
        data_split (int, optional): Index to split the data into a train set and a test set. Defaults to None.
            data_split must be specified if estimator is instantiated but not fitted.
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

    numeric_data = data.select_dtypes("number")

    # ensures date_col is a datetime object and sets as datetimeindex
    if date_col:
        if date_col != "index":
            numeric_data.index = pd.to_datetime(data[date_col])
            if not numeric_data.index.is_monotonic_increasing:
                numeric_data.sort_index(inplace=True)

    anomalywidget = _get_compute_backend(compute_backend, numeric_data).compute_anomaly(
        data=numeric_data,
        target=target,
        date_col=date_col,
        estimator=estimator,
        n_periods=n_periods,
        data_split=data_split,
        method=method,
        **kwargs,
    )

    anomalywidget.viz_backend = viz_backend
    anomalywidget.date_col = date_col
    anomalywidget.target = target
    anomalywidget.viz_backend = viz_backend
    anomalywidget.n_periods = n_periods
    anomalywidget.sigma = sigma

    return anomalywidget


def _pandas_compute_anomaly(
    data,
    target: Optional[str] = None,
    date_col: Optional[str] = None,
    method: Optional[str] = "arima",
    estimator=None,
    n_periods: Optional[int] = None,
    data_split: Optional[int] = None,
    **kwargs,
):
    """Backend implementation of anomaly detection.

    Args:
        data (DataFrame): The dataframe.
        target (str, optional): The target column. Defaults to None. If target is None, unsupervised methods are used.
        date_col (str, optional): Datetime column if data is timeseries. Defaults to None. If data is timeseries, date_col must be specified.
        method (str, optional): Select method from this list. Only "arima" is supported.
        estimator (optional): Fitted or instantiated estimator with a .predict() and .fit() method. Defaults to None.
            If estimator is instantiated but not fitted, data_split must be specified.
        n_periods (int, optional): Number of periods for timeseries anomaly detection. Default is None.
        data_split (int, optional): Index to split the data into a train set and a test set. Defaults to None.
            data_split must be specified if estimator is instantiated but not fitted.
        **kwargs: Keyword arguments.

    Raises:
        ValueError: If method is not implemented.

    Returns:
        AnomalyDetectionWidget

    """
    # Timeseries indicator
    if date_col:

        # Supervised learning indicator
        if target:
            train, test = (
                data[target][0:data_split],
                data[target][data_split:],
            )

            # Default to ARIMA model
            if not estimator:
                from pmdarima.arima import auto_arima

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
            predictions_df = stepwise_fit_and_predict(
                train=train, test=test, n_periods=n_periods, estimator=estimator
            )

            # Post-processing for errors and confidence interval
            predictions_df = _pandas_compute_anomalies_stats(
                predictions_df, n_periods=n_periods
            )

        # Indicator for unsupervised learning
        else:
            raise ValueError(
                "Unsupervised timeseries methods for anomaly detection are not yet supported."
            )

    # Indicator for regression and classification learning
    else:
        raise ValueError(
            "Regression and Classification methods for anomaly detection are not yet supported."
        )

    return AnomalyDetectionWidget(
        estimator=estimator,
        method=method,
        viz_data=predictions_df,
        data_split=data_split,
    )


def stepwise_fit_and_predict(train, test, n_periods, estimator):
    """Perform stepwise fit and predict for timeseries data.

    Args:
        train (DataFrame): The training data.
        test (DataFrame): The testing data.
        n_periods (int): The number of periods.
        estimator: The estimator.

    Returns:
        predictions_df: DataFrame containing the ground truth, predictions, and indexed by the datetime.
    """
    history = [x for x in train]
    predictions = list()
    for t in test.index:
        estimator.fit(history)
        output = estimator.predict(n_periods=n_periods)
        predictions.append(output[0])
        obs = test[t]
        history.append(obs)

    predictions_df = pd.DataFrame()
    predictions_df["actuals"] = test
    predictions_df["predictions"] = predictions
    return predictions_df


def _pandas_compute_anomalies_stats(predictions_df, n_periods, sigma=2):
    """Detects anomalies based on the statistical profiling of the residuals (actuals - predicted).

    The rolling mean and rolling standard deviation is used to identify points that are more than 2 standard deviations away from the mean.

    Args:
        predictions_df (DataFrame): The dataframe containing the ground truth and predictions.
        n_periods (int, optional): The number of periods.
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
    predictions_df["meanval"] = predictions_df["error"].rolling(window=n_periods).mean()
    predictions_df["deviation"] = (
        predictions_df["error"].rolling(window=n_periods).std()
    )
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
        marker_color (str): The color to mark anomalies. Defaults to "red".

    Returns:
        Plotly plot

    """
    lookback = -1 * (n_periods - 1)
    predictions_df = predictions_df.iloc[:lookback, :]
    # predictions_df.reset_index(inplace=True)
    bool_array = abs(predictions_df["anomaly_points"]) > 0
    actuals = predictions_df["actuals"][-len(bool_array) :]
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan

    # color_map = {0: "'rgba(228, 222, 249, 0.65)'", 1: "yellow", 2: "orange", 3: "red"}

    anomalies = go.Scatter(
        name="Predicted Anomaly",
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
        # showlegend=False,
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
        name="Actual Anomaly",
        # showlegend=False,
        x=predictions_df.index,
        y=anomaly_points,
        mode="markers",
        xaxis="x2",
        yaxis="y2",
        marker=dict(color="red", size=11, line=dict(color="red", width=2)),
    )

    MovingAverage = go.Scatter(
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
            MovingAverage,
            Error,
        ],
        layout=layout,
    )

    if _in_notebook():
        po.init_notebook_mode(connected=True)
        return po.iplot(fig)
    else:
        return fig
