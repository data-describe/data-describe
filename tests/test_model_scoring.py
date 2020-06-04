import pytest
import pandas as pd
import numpy as np
import matplotlib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from mwdata.modeling.score import (
    metric_table,
    confusion_matrix,
    roc_curve_plot,
    prediction_distribution_plot,
    pr_curve_plot,
)


def test_metric_table_default():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0])

    with pytest.warns(UserWarning):
        metrics = metric_table(y_true, y_pred)
        assert metrics.shape == (1, 5)
        assert metrics.iloc[0, 0] == accuracy_score(y_true, y_pred)
        assert metrics.iloc[0, 1] == precision_score(y_true, y_pred)
        assert metrics.iloc[0, 2] == recall_score(y_true, y_pred)
        assert metrics.iloc[0, 3] == f1_score(y_true, y_pred)
        assert metrics.iloc[0, 4] == roc_auc_score(y_true, y_pred)


def test_metric_table_proba():
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred_proba = np.array([0.2, 0.77, 0.15, 0.98, 0.46])

    metrics = metric_table(y_true, y_pred_proba)
    assert metrics.shape == (1, 5)
    assert metrics.iloc[0, 0] == accuracy_score(y_true, y_pred_proba > 0.5)
    assert metrics.iloc[0, 1] == precision_score(y_true, y_pred_proba > 0.5)
    assert metrics.iloc[0, 2] == recall_score(y_true, y_pred_proba > 0.5)
    assert metrics.iloc[0, 3] == f1_score(y_true, y_pred_proba > 0.5)
    assert metrics.iloc[0, 4] == roc_auc_score(y_true, y_pred_proba)


def test_metric_table_list():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    metrics = metric_table(y_true, y_pred)
    assert metrics.shape == (1, 5)


def test_metric_table_pandas_series():
    y_true = pd.Series([0, 1, 0, 1, 0])
    y_pred = pd.Series([1, 1, 0, 1, 0])
    metrics = metric_table(y_true, y_pred)
    assert metrics.shape == (1, 5)


def test_metric_table_pandas_dataframe():
    y_true = pd.Series([0, 1, 0, 1, 0])
    y_pred = pd.DataFrame({"modelA": [1, 1, 0, 1, 0], "modelB": [0, 0, 0, 1, 0]})
    metrics = metric_table(y_true, y_pred)
    assert metrics.shape == (2, 5)


def test_metric_table_unsupported():
    y_true = {"a": [0, 1, 0, 1, 0]}
    y_pred = {"b": [0, 1, 0, 1, 0]}
    with pytest.raises(ValueError):
        metric_table(y_true, y_pred)


def test_confusion_matrix():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    cm = confusion_matrix(y_true, y_pred)
    assert isinstance(cm, matplotlib.artist.Artist)


def test_normalized_confusion_matrix():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    cm = confusion_matrix(y_true, y_pred, normalize=True)
    assert isinstance(cm, matplotlib.artist.Artist)


def test_interactive_confusion_matrix():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    with pytest.raises(NotImplementedError):
        confusion_matrix(y_true, y_pred, interactive=True)


def test_roc_plot():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    assert roc_curve_plot(y_true, y_pred) is None


def test_roc_plot_df():
    y_true = [0, 1, 0, 1, 0]
    y_pred = pd.DataFrame({"A": [1, 1, 0, 1, 0], "B": [0.88, 0.11, 0.24, 0.24, 0.58]})
    assert roc_curve_plot(y_true, y_pred) is None


def test_interactive_roc_plot():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    with pytest.raises(NotImplementedError):
        roc_curve_plot(y_true, y_pred, interactive=True)


def test_pr_plot():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    assert pr_curve_plot(y_true, y_pred) is None


def test_interactive_pr_plot():
    y_true = [0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 1, 0]
    with pytest.raises(NotImplementedError):
        pr_curve_plot(y_true, y_pred, interactive=True)


def test_pr_plot_df():
    y_true = [0, 1, 0, 1, 0]
    y_pred = pd.DataFrame({"A": [1, 1, 0, 1, 0], "B": [0.88, 0.11, 0.24, 0.24, 0.58]})
    assert pr_curve_plot(y_true, y_pred) is None


def test_pp_plot_list():
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [1, 1, 0, 1, 0, 0]
    assert isinstance(
        prediction_distribution_plot(y_true, y_pred), matplotlib.artist.Artist
    )


def test_pp_plot_np():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    assert isinstance(
        prediction_distribution_plot(y_true, y_pred), matplotlib.artist.Artist
    )


def test_pp_plot_df():
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_pred = pd.Series([1, 1, 0, 1, 0, 0])
    assert isinstance(
        prediction_distribution_plot(y_true, y_pred), matplotlib.artist.Artist
    )


def test_pp_plot_fails():
    with pytest.raises(ValueError):
        prediction_distribution_plot(0, 1)
