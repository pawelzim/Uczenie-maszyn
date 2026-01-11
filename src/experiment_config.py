from pathlib import Path


def build_suffix(names):
    if names is None:
        return ""
    names = sorted(set(names))
    return "_" + "_".join(names) if names else ""


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results" / "tables"

ENABLED_CLASSIFIERS = None
ENABLED_REDUCERS = ["lda"]
# np
# ENABLED_CLASSIFIERS = ["svm_rbf", "mlp", "rf", "logreg", "knn"]
# ENABLED_REDUCERS = ["pca", "lda", "kpca", "ica"]

SYNTHETIC_EXPERIMENT = "E1"  # "E1" "E2"
