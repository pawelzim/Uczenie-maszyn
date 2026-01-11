import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_generation import experiment2_imbalance_configs, generate_synthetic_dataset
from datasets import load_real_dataset
from models import get_classifiers
from metrics import compute_metrics
from experiment_config import *


def evaluate_pipeline_cv(X, y, pipeline, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for train_idx, test_idx in skf.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        pred = pipeline.predict(X[test_idx])
        rows.append(compute_metrics(y[test_idx], pred))

    df = pd.DataFrame(rows)
    return df.mean().to_dict(), df.std().to_dict()


def main():
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_configs = experiment2_imbalance_configs()
    syn_data = [(cfg.name, *generate_synthetic_dataset(cfg)) for cfg in syn_configs]

    real_names = ["breast_cancer", "iris"]
    real_data = [(name, *load_real_dataset(name)) for name in real_names]

    datasets = syn_data + real_data

    classifiers_all = get_classifiers()
    classifiers = (
        classifiers_all
        if ENABLED_CLASSIFIERS is None
        else {k: v for k, v in classifiers_all.items() if k in ENABLED_CLASSIFIERS}
    )

    rows = []
    for ds_name, X, y, meta in datasets:
        for clf_name, clf in classifiers.items():
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ]
            )

            mean, std = evaluate_pipeline_cv(X, y, pipe)

            rows.append(
                {
                    "dataset": ds_name,
                    "classifier": clf_name,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )

    out = pd.DataFrame(rows)
    classifiers_suffix = build_suffix(ENABLED_CLASSIFIERS)
    out.to_csv(out_dir / f"baseline_cv{classifiers_suffix}.csv", index=False)


if __name__ == "__main__":
    main()
