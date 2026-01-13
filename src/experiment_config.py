from pathlib import Path


def build_suffix(names):
    if names is None:
        return ""
    names = sorted(set(names))
    return "_" + "_".join(names) if names else ""


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results" / "tables"

ENABLED_CLASSIFIERS = ["knn", "svm_rbf", "mlp"]
# Wszystkie klasyfikatory: ["knn", "svm_rbf", "mlp"]

ENABLED_REDUCERS = ["ica"]
# Wszystkie metody redukcji: ["pca", "kpca", "ica", "lda"] ; lda pomijamy

SYNTHETIC_EXPERIMENT = "E2"  # "E1" "E2"
