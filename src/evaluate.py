import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error, r2_score
)

def evaluate(task: str, y_true, y_pred, y_proba=None):
    if task == "classification":
        m = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_proba is not None:
            try:
                m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                m["roc_auc"] = None
        return m

    if task == "regression":
        rmse = root_mean_squared_error(y_true, y_pred)
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(rmse),
            "r2": float(r2_score(y_true, y_pred)),
        }
    raise ValueError("Unknown task")
