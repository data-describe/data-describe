import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix as sk_cm

from mwdata.utilities.contextmanager import _context_manager


def metric_table(y_true, y_pred, name=None, threshold=0.5):
    """ Summary table of model metrics

    Args:
        y_true: The response / labels
        y_pred: The predictions. If values are floats, assumed to be prediction probabilities.
        y_pred can also be a Pandas Data Frame, where each column contains the predictions for a different model.
        name: A display name for the model
        threshold: The default decision threshold

    Returns:
        Pandas dataframe of model metrics
    """
    if isinstance(y_pred, np.ndarray):
        pass
    elif isinstance(y_pred, pd.Series):
        name = y_pred.name
        y_pred = y_pred.to_numpy()
    elif isinstance(y_pred, pd.DataFrame):
        return pd.concat(
            [metric_table(y_true, x[1]) for x in y_pred.iteritems()], axis=0
        )
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    else:
        raise ValueError("Unsupported data type for `y_pred`")

    if name is None:
        name = 0

    if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
        y_pred_proba = y_pred
        y_pred = y_pred > threshold
        auc = roc_auc_score(y_true, y_pred_proba)
    else:
        warnings.warn("AUC cannot be computed accurately without probabilities.")
        auc = roc_auc_score(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return pd.DataFrame(
        {
            "Accuracy": [acc],
            "Precision": [prec],
            "Recall": [rec],
            "F1": [f1],
            "AUC": [auc],
        },
        index=[name],
    )


@_context_manager
def roc_curve_plot(y_true, y_pred_proba, interactive=False, context=None):
    """ Creates a ROC curve plot for one or more models

    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true: The response / labels
        y_pred_proba: The prediction probabilities
        y_pred_proba can also be a Pandas Data Frame, where each column contains the prediction probabilities for a different model.
        interactive: If True, returns an interactive plot, using Plotly
        context: The context

    Returns:
        The ROC Plot
    """
    if interactive:
        raise NotImplementedError
    else:
        plt.figure(figsize=(context.fig_width, context.fig_height))
        line_width = 2
        if isinstance(y_pred_proba, pd.DataFrame):
            for m in y_pred_proba.columns.values:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[m])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    lw=line_width,
                    label="{0} (area = {1:0.2f})".format(m, roc_auc),
                )
        else:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                lw=line_width,
                label="{0} (area = {1:0.2f})".format("Model", roc_auc),
            )
        plt.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=line_width,
            linestyle="--",
            label="Baseline (Random)",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")


@_context_manager
def pr_curve_plot(y_true, y_pred_proba, interactive=False, context=None):
    """ Creates a precision-recall curve plot for one or more models

    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true: The response / labels
        y_pred_proba: The prediction probabilities
        y_pred_proba can also be a Pandas Data Frame, where each column contains the prediction probabilities for a different model.
        interactive: If True, returns an interactive plot, using Plotly
        context: The context

    Returns:
        The PR Curve Plot
    """
    if interactive:
        raise NotImplementedError
    else:
        plt.figure(figsize=(context.fig_width, context.fig_height))
        line_width = 2
        if isinstance(y_pred_proba, pd.DataFrame):
            for m in y_pred_proba.columns.values:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[m])
                plt.plot(recall, precision, lw=line_width, label=m)
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            plt.plot(recall, precision, lw=line_width, label="Model")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower right")


@_context_manager
def confusion_matrix(
    y_true, y_pred, normalize=False, cmap=plt.cm.Blues, interactive=False, context=None
):
    """ Creates a confusion matrix visualization

    Args:
        y_true: The response / labels
        y_pred: The predictions
        normalize: If True, normalize predictions across True Labels
        cmap: The Matplotlib colormap for the plot
        interactive: If True, create an interactive plot, using Plotly
        context: The context

    Returns:
        Confusion Matrix plot
    """
    cm = sk_cm(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
    else:
        title = "Confusion Matrix"

    if interactive:
        raise NotImplementedError(
            "Interactive plot for Confusion Matrix is not implemented."
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        image = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(image, ax=ax)
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xlim=[-0.5, cm.shape[1] - 0.5],
            ylim=[-0.5, cm.shape[0] - 0.5],
            title=title,
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        num_format = ".2f" if normalize else "d"
        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], num_format),
                    ha="center",
                    va="center",
                    size=20,
                    color="white" if cm[i, j] > threshold else "black",
                )
        fig.tight_layout()
        return ax


@_context_manager
def prediction_distribution_plot(y_true, y_pred_proba, true_label=1, context=None):
    """Prediction probability distributions for each class in the response

    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true: The response / labels
        y_pred_proba: The prediction probabilities
        true_label: The value in `y_true` corresponding to the "True" label
        context: The context

    Returns:

    """
    if isinstance(y_true, list):
        idx_true = [i for i, x in enumerate(y_true) if x == true_label]
        idx_false = [i for i, x in enumerate(y_true) if x != true_label]
        true_distribution = [y_pred_proba[i] for i in idx_true]
        false_distribution = [y_pred_proba[i] for i in idx_false]
    elif isinstance(y_true, pd.Series):
        idx = y_true == true_label
        true_distribution = y_pred_proba[idx]
        false_distribution = y_pred_proba[~idx]
    elif isinstance(y_true, np.ndarray):
        idx = y_true == true_label
        true_distribution = y_pred_proba[idx]
        false_distribution = y_pred_proba[~idx]
    else:
        raise ValueError("Unrecognized data type for `y_true`")

    plt.figure(figsize=(context.fig_width, context.fig_height))
    ax = sns.kdeplot(true_distribution, shade=True, label="True")
    ax = sns.kdeplot(false_distribution, shade=True, label="False", ax=ax)
    plt.title("Prediction Distributions")
    plt.xlim([0.0, 1.0])
    plt.xlabel("Probability of Event")
    plt.ylabel("Probability Density")
    return ax
