import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from data_generation import (
    experiment1_feature_configs,
    experiment2_imbalance_configs,
    generate_synthetic_dataset,
)
from datasets import load_real_dataset
from models import get_classifiers
from reducers import get_reducers
from metrics import compute_metrics
from experiment_config import *


def evaluate_pipeline_cv(X, y, pipeline, n_splits=4, random_state=42):
    # foldowanie datasetow do walidacji krzyzowej, potrzebne jak nie ma osobnych zbiorow do uczenia i testowania
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for _, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # klonowanie pipelineu przed kazda iteracja, bo potencjalnie kopiowalo i wstawialo identyczne wyniki
        pipe = clone(pipeline)

        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        rows.append(compute_metrics(y[test_idx], pred))

    df = pd.DataFrame(rows)
    return df.mean(numeric_only=True).to_dict(), df.std(numeric_only=True).to_dict()


def main():
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # wybor syntetycznego zbioru eksperymentalnego + rzeczywistego
    if SYNTHETIC_EXPERIMENT == "E1":
        syn_configs = experiment1_feature_configs(n_samples=800)
        exp_tag = "_E1"
        real_names = ["breast_cancer", "wine", "digits"]
    elif SYNTHETIC_EXPERIMENT == "E2":
        syn_configs = experiment2_imbalance_configs()
        exp_tag = "_E2"
        real_names = []
    else:
        raise ValueError("Incorrect SYNTHETIC_EXPERIMENT, should be E1, E2")

    syn_data = [(cfg.name, *generate_synthetic_dataset(cfg)) for cfg in syn_configs]
    real_data = [(name, *load_real_dataset(name)) for name in real_names]
    datasets = syn_data + real_data

    classifiers_all = get_classifiers(random_state=42)
    classifiers = (
        classifiers_all
        if ENABLED_CLASSIFIERS is None
        else {k: v for k, v in classifiers_all.items() if k in ENABLED_CLASSIFIERS}
    )

    rows = []
    for ds_name, X, y, _ in datasets:
        n_features = X.shape[1]

        if n_features <= 10:
            n_components_list = [max(2, n_features - 1)]

        elif n_features < 300:
            n_components_list = [
                max(3, int(n_features * 0.30)),
                max(3, int(n_features * 0.50)),
                max(3, int(n_features * 0.70)),
            ]

        else:
            n_components_list = [20, 50, 100]

        n_components_list = sorted({nc for nc in n_components_list if nc < n_features})

        print(f"Processing {ds_name} (baseline)...")
        for clf_name, clf in classifiers.items():
            pipe_none = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", clone(clf)),
                ]
            )
            mean, std = evaluate_pipeline_cv(X, y, pipe_none)

            rows.append(
                {
                    "dataset": ds_name,
                    "reducer": "none",
                    "n_components": None,
                    "classifier": clf_name,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )

        for n_components in n_components_list:
            print(f"Processing {ds_name} (n_components={n_components})...")
            gamma = 1.0 / X.shape[1]
            reducers_all = get_reducers(
                n_components, n_classes=len(np.unique(y)), kpca_gamma=gamma
            )
            reducers_all = {k: v for k, v in reducers_all.items() if k != "ccpca"}
            reducers = (
                reducers_all
                if ENABLED_REDUCERS is None
                else {k: v for k, v in reducers_all.items() if k in ENABLED_REDUCERS}
            )

            for red_name, reducer in reducers.items():
                for clf_name, clf in classifiers.items():
                    pipe = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("reducer", clone(reducer)),
                            ("clf", clone(clf)),
                        ]
                    )
                    mean, std = evaluate_pipeline_cv(X, y, pipe)

                    rows.append(
                        {
                            "dataset": ds_name,
                            "reducer": red_name,
                            "n_components": n_components,
                            "classifier": clf_name,
                            **{f"{k}_mean": v for k, v in mean.items()},
                            **{f"{k}_std": v for k, v in std.items()},
                        }
                    )

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "n_components", "reducer", "classifier"]
    )
    reducers_suffix = build_suffix(ENABLED_REDUCERS)
    out.to_csv(out_dir / f"reduction_cv{exp_tag}{reducers_suffix}.csv", index=False)


if __name__ == "__main__":
    main()
