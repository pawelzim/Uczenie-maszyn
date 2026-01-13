from typing import Dict, Tuple
import numpy as np

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits


def load_real_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Return X, y oraz metadane datasetu
    "breast_cancer", "iris", "wine", "digits" z datasetow rzeczywistycj
    """
    name = name.lower().strip()

    if name == "breast_cancer":
        ds = load_breast_cancer()
        X, y = ds.data, ds.target
        meta = {
            "dataset": "breast_cancer",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
        return X, y, meta

    if name == "iris":
        ds = load_iris()
        X, y = ds.data, ds.target
        meta = {
            "dataset": "iris",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
        return X, y, meta

    if name == "wine":
        ds = load_wine()
        X, y = ds.data, ds.target
        meta = {
            "dataset": "wine",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
        return X, y, meta

    if name == "digits":
        ds = load_digits()
        X, y = ds.data, ds.target
        meta = {
            "dataset": "digits",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }
        return X, y, meta

    raise ValueError(f"Unknown dataset name: {name}. breast_cancer, iris, wine, digits")
