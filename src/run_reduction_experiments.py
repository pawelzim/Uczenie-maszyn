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


# walidacja 2 foldowa 5 krotna
# testy statystyczne
# opisy metody
# batch size manipulacja
# max iter w kpca mozna zredukowac


def evaluate_pipeline_cv(X, y, pipeline, random_state=42):
    rows = []

    for r in range(5):
        skf = StratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=random_state + r,
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            pipe = clone(pipeline)
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[test_idx])

            m = compute_metrics(y[test_idx], pred)
            rows.append({"repeat": r, "fold": fold, **m})

    raw_df = pd.DataFrame(rows)
    mean = raw_df.drop(columns=["repeat", "fold"]).mean(numeric_only=True).to_dict()
    std = raw_df.drop(columns=["repeat", "fold"]).std(numeric_only=True).to_dict()
    return mean, std, raw_df


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
        real_names = ["breast_cancer", "wine", "digits"]
    else:
        raise ValueError("Incorrect SYNTHETIC_EXPERIMENT, should be E1, E2")

    # ladowanie datasetow syn + real
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
    raw_rows = []

    for ds_name, X, y, _ in datasets:
        n_features = X.shape[1]

        # 'adadptacyjny' dobor liczby komponenetow redukcji
        if n_features <= 10:
            n_components_list = [max(2, n_features - 1)]
        elif n_features < 200:
            n_components_list = [
                max(3, int(n_features * 0.30)),
                max(3, int(n_features * 0.50)),
                max(3, int(n_features * 0.70)),
            ]
        else:
            n_components_list = [20, 50, 100]

        n_components_list = sorted({nc for nc in n_components_list if nc < n_features})

        # baseline
        print(f"Processing {ds_name} (baseline)...")
        for clf_name, clf in classifiers.items():
            pipe_none = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", clone(clf)),
                ]
            )
            mean, std, raw_df = evaluate_pipeline_cv(X, y, pipe_none)

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
            raw_df = raw_df.copy()
            raw_df["dataset"] = ds_name
            raw_df["classifier"] = clf_name
            raw_df["reducer"] = "none"
            raw_df["n_components"] = np.nan
            raw_rows.append(raw_df)

        # z redukcja
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
                    mean, std, raw_df = evaluate_pipeline_cv(X, y, pipe)

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
                    raw_df = raw_df.copy()
                    raw_df["dataset"] = ds_name
                    raw_df["classifier"] = clf_name
                    raw_df["reducer"] = red_name
                    raw_df["n_components"] = n_components
                    raw_rows.append(raw_df)

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "n_components", "reducer", "classifier"]
    )
    reducers_suffix = build_suffix(ENABLED_REDUCERS)
    out.to_csv(out_dir / f"reduction_cv{exp_tag}{reducers_suffix}.csv", index=False)
    raw_out = pd.concat(raw_rows, ignore_index=True)
    raw_out = raw_out.sort_values(
        ["dataset", "classifier", "reducer", "n_components", "repeat", "fold"]
    )
    raw_out.to_csv(
        out_dir / f"reduction_cv_raw{exp_tag}{reducers_suffix}.csv", index=False
    )


if __name__ == "__main__":
    main()
