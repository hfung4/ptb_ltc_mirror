from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def generate_test_metrics(
    y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Generate various test metrics for the model

    Args:
        y_test (pd.Series): label for the test set
        y_pred (np.ndarray): test predictions
        y_pred_proba (np.ndarray): test probability score

    Returns:
        Dict:a dictionary containing the test metrics
    """

    test_log_loss = log_loss(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0.0)
    test_recall = recall_score(y_test, y_pred, zero_division=0.0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0.0)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    test_auprc = average_precision_score(y_test, y_pred_proba[:, 1])
    test_auprc_baseline = y_test.mean()
    test_auprc_lift = test_auprc / test_auprc_baseline if test_auprc_baseline > 0 else 0
    test_brier = brier_score_loss(y_test, y_pred_proba[:, 1])


    return {
        "test_log_loss": test_log_loss,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_auprc": test_auprc,
        "test_auprc_baseline": test_auprc_baseline,
        "test_auprc_lift": test_auprc_lift,
        "test_brier": test_brier,
    }
