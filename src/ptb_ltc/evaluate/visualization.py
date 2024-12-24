import itertools
from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LearningCurveDisplay, learning_curve

from ptb_ltc.config.core import config


def generate_prc_plot(
    y_test: pd.Series,
    y_scores: np.ndarray,
    mapping: OrderedDict[int, str] = OrderedDict({0: "negative", 1: "positive"}),
    figsize: Tuple = (10, 5),
) -> Figure:
    """Generates the test precision and recall plot

    Args:
        y_test (pd.Series): test set labels
        y_scores (np.ndarray): probability scores
        mapping (OrderedDict[int,str]): mapping of numeric to string labels for the outcome variable. Defaults to OrderedDict({0: "negative", 1: "positive"}).
        figsize (tuple, optional): _description_. Defaults to (20, 8).

    Returns:
        fig: output figure
    """

    # structures
    precisions = dict()
    recalls = dict()
    thresholds = dict()
    auprcs = dict()

    # get dummy variables for y_test, one for each level
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values

    # Compute precision, recall for each class
    # Also compute the AUPRC for each class
    for k, v in mapping.items():
        precisions[v], recalls[v], _ = precision_recall_curve(
            y_test_dummies[:, k], y_scores[:, k]
        )
        auprcs[v] = average_precision_score(y_test_dummies[:, k], y_scores[:, k])

    # plot precision and recall vs threshold for each class
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use("fivethirtyeight")
    plt.rcParams["font.size"] = 12

    for _, v in mapping.items():
        if v == "positive":
            baseline = y_test.mean()
        else:
            baseline = 1 - y_test.mean()
        ax.plot(
            recalls[v],
            precisions[v],
            label=f"PRC for {v} class (area = {round(auprcs[v],2)}, baseline = {round(baseline,2)})",
        )

    # Plot settings
    ax.set_xlim([0.0, 1.0])  # set x and y limits
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel("Recall")  # set x and y labels and title
    ax.set_ylabel("Precision")
    ax.set_title("Precision and Recall Curves")
    ax.legend(loc="best")
    fig.tight_layout()

    plt.close(fig)

    return fig


def generate_roc_plot(
    y_test: pd.Series,
    y_scores: np.ndarray,
    mapping: OrderedDict[int, str] = OrderedDict({0: "negative", 1: "positive"}),
    figsize: Tuple = (10, 5),
) -> Figure:
    """Generates the test ROC plot

    Args:
        y_test (pd.Series): test set labels
        y_scores (np.ndarray): probability scores
        mapping (OrderedDict[int,str]): mapping of numeric to string labels for the outcome variable. Defaults to OrderedDict({0: "negative", 1: "positive"}).
        figsize (tuple, optional): _description_. Defaults to (20, 8).

    Returns:
        fig: output figure
    """
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # get dummy variables for y_test, one for each level
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values

    # Compute fpr and tpr for each class
    # Also compute the ROC_AUC for each class
    for k, v in mapping.items():
        fpr[v], tpr[v], _ = roc_curve(y_test_dummies[:, k], y_scores[:, k])
        roc_auc[v] = roc_auc_score(y_test_dummies[:, k], y_scores[:, k])

    # plot roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use("fivethirtyeight")
    plt.rcParams["font.size"] = 12

    for _, v in mapping.items():
        ax.plot(
            fpr[v],
            tpr[v],
            label=f"ROC curve for {v} class (area = {round(roc_auc[v],2)})",
        )

    # plot settings
    ax.plot([0, 1], [0, 1], "k--")  # plot the 45 deg line

    ax.set_xlim([0.0, 1.0])  # set x and y limits
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel("False Positive Rate")  # set x and y labels and title
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROCs")
    ax.legend(loc="best")
    fig.tight_layout()

    plt.close(fig)
    return fig


def generate_confusion_matrix_plot(
    y_test: pd.Series,
    y_pred: np.ndarray,
    classes: List[str],
    normalize: bool = True,
    title: str = "Confusion matrix",
    cmap: object = plt.cm.Oranges,
    figsize: Tuple = (10, 5),
) -> Figure:
    """Generates the test confusion matrix plot

    Args:
        y_test (pd.Series): test set labels
        y_pred (np.ndarray): predicted labels
        classes (List[str]): names of the classes of the label
        normalize (bool, optional): plot normalized confusion matrix. Defaults to True.
        title (str, optional): title of the confusion matrix plot. Defaults to "Confusion matrix".
        cmap (object, optional): heatmap color for the confusion matrix plot. Defaults to plt.cm.Oranges.
        figsize (Tuple, optional): size of the confusion matrix plot. Defaults to (10, 5).

    Returns:
        fig: output figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize confusion matrix
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, size=16)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    # Label the confusion matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            fontsize=12,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel("True label", size=12)
    plt.xlabel("Predicted label", size=12)
    fig.tight_layout()

    plt.close(fig)
    return fig


def generate_shapley_values_plot(
    model: object, X_processed: pd.DataFrame, figsize: Tuple = (10, 5)
) -> Figure:
    """Computes the SHAP value and generates shapley values bar plot

    Args:
        model (object): optimized model
        X_processed (pd.DataFrame): processed features DataFrame, can be from train or test set
        figsize (Tuple, optional): size of the plot. Defaults to (20, 8).

    Returns:
        fig: output figure
    """
    explainer = shap.Explainer(model, X_processed)
    shap_values = explainer.shap_values(X_processed)

    # Create a mask for rows that do not exceed the outlier threshold
    # This example filters out any row that has an absolute SHAP value > threshold
    mask = (np.abs(shap_values) < config.model.SHAP_THRESHOLD).all(axis=1)

    # Apply the mask to both the feature matrix and shap values
    X_filtered = X_processed[mask]
    shap_values_filtered = shap_values[mask]

    # Now, create the summary plot without the extreme outliers
    plot_type = None if config.model.SHAP_TYPE == 'beeswarm' else config.model.SHAP_TYPE
    shap.summary_plot(shap_values_filtered, X_filtered, plot_size=figsize, show=False, plot_type=plot_type)

    fig = plt.gcf()
    fig.tight_layout()
    plt.close(fig)
    return fig


def generate_learning_curves(
    pipeline: object,
    scorer: object,
    score_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    figsize: Tuple = (10, 5),
    train_sizes: List = [0.2, 0.4, 0.6, 0.8, 1.0],
) -> Figure:
    """Generates learning curves for the model

    Args:
        pipeline (object): optimized data pipeline that contains processing steps and the model
        scorer (object): sklearn scorer object
        score_name (str): name of the performance metric, will be used to label the plot
        X_train (pd.DataFrame): feature dataframe from the train set
        y_train (pd.Series): ou
        train_sizes (List, optional): _description_. Defaults to [0.2, 0.4, 0.6, 0.8, 1.0].

    Returns:
        f: output figure
    """
    # Disable mflow.autolog because we will be fitting models with different train size
    # and fit() will trigger mlflow autolog
    mlflow.autolog(disable=True)

    # Compute train and validation scores at different train sizes
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=config.model.N_FOLDS,
        scoring=scorer,
        shuffle=True,
        random_state=config.general.RANDOM_STATE,
        train_sizes=train_sizes,
    )

    # Plot learning curve
    fig, ax = plt.subplots(figsize=figsize)
    display = LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=valid_scores,
        score_name=score_name,
    )
    display.plot(ax=ax)

    # Re-enable mlflow autlog
    mlflow.autolog(disable=False)

    f = plt.gcf()
    f.tight_layout()
    plt.close(f)
    return f


def generate_calibration_curve(
    y_test: pd.Series, y_pred_proba: np.ndarray, figsize: Tuple = (15, 8)
) -> Figure:
    """Generates the calibration curve for the model

    Args:
        y_test (pd.Series): test labels
        y_pred_proba (np.ndarray): test set probability scores
        figsize (Tuple, optional): size of the calibration curve. Defaults to (20, 8).

    Returns:
        f: output figure
    """
    # Compute the true and predicted probabilities
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)

    # Plot the calibration curve
    fig, ax = plt.subplots(figsize=figsize)
    disp = CalibrationDisplay(prob_true, prob_pred, y_pred_proba[:, 1])
    disp.plot(ax=ax)

    f = plt.gcf()
    f.tight_layout()
    plt.close(f)
    return f
