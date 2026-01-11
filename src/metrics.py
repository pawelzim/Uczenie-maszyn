from typing import Dict
import numpy as np
from sklearn.metrics import (
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)

    classes = np.unique(y_true)
    avg = "binary" if len(classes) == 2 else "macro"

    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)

    gmean_pdf = float(np.sqrt(prec * rec))

    return {
        "accuracy": float(acc),
        "bac": float(bac),
        "precision": float(prec),
        "recall": float(rec),
        "gmean_pdf": float(gmean_pdf),
    }
