from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from experiment_config import RESULTS_DIR

METRICS = ["accuracy", "bac", "precision", "recall", "gmean_pdf"]


def _pick_latest_raw_file(tables_dir: Path) -> Path:
    files = sorted(
        tables_dir.glob("reduction_cv_raw_E2_kpca.csv"), key=lambda p: p.stat().st_mtime
    )
    if not files:
        raise FileNotFoundError(
            f"Nie znaleziono pliku raw: {tables_dir}/reduction_cv_raw*.csv. "
            f"Najpierw uruchom run_reduction_experiments.py z zapisem per-fold."
        )
    return files[-1]


def _validate_columns(df: pd.DataFrame) -> None:
    required = {
        "dataset",
        "classifier",
        "reducer",
        "n_components",
        "repeat",
        "fold",
    } | set(METRICS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w raw CSV: {sorted(missing)}")


def _paired_merge(
    df: pd.DataFrame,
    *,
    dataset: str,
    classifier: str,
    reducer: str,
    n_components: Optional[int],
) -> pd.DataFrame:
    base = df[
        (df["dataset"] == dataset)
        & (df["classifier"] == classifier)
        & (df["reducer"] == "none")
    ].copy()

    red = df[
        (df["dataset"] == dataset)
        & (df["classifier"] == classifier)
        & (df["reducer"] == reducer)
        & (
            df["n_components"].isna()
            if n_components is None
            else df["n_components"] == n_components
        )
    ].copy()

    key = ["dataset", "classifier", "repeat", "fold"]
    base = base[key + METRICS].rename(columns={m: f"{m}_base" for m in METRICS})
    red = red[key + METRICS].rename(columns={m: f"{m}_red" for m in METRICS})
    return pd.merge(base, red, on=key, how="inner")


def _run_tests_for_metric(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float]:
    # paired t-test
    t_stat, t_p = ttest_rel(y, x, nan_policy="omit")

    diff = y - x
    if np.allclose(diff, 0.0, equal_nan=True):
        return float(t_stat), float(t_p), np.nan, 1.0

    w = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return float(t_stat), float(t_p), float(w.statistic), float(w.pvalue)


def main(raw_path: Optional[str] = None, out_path: Optional[str] = None) -> None:
    raw_file = Path(raw_path) if raw_path else _pick_latest_raw_file(RESULTS_DIR)
    df = pd.read_csv(raw_file)

    _validate_columns(df)
    df["n_components"] = pd.to_numeric(df["n_components"], errors="coerce")

    configs = (
        df[df["reducer"] != "none"][
            ["dataset", "classifier", "reducer", "n_components"]
        ]
        .drop_duplicates()
        .sort_values(["dataset", "classifier", "reducer", "n_components"])
        .to_records(index=False)
    )

    rows: List[dict] = []

    for dataset, classifier, reducer, n_components in configs:
        merged = _paired_merge(
            df,
            dataset=str(dataset),
            classifier=str(classifier),
            reducer=str(reducer),
            n_components=None if pd.isna(n_components) else int(n_components),
        )

        if len(merged) < 2:
            continue

        for m in METRICS:
            x = merged[f"{m}_base"].to_numpy(dtype=float)
            y = merged[f"{m}_red"].to_numpy(dtype=float)

            t_stat, t_p, w_stat, w_p = _run_tests_for_metric(x, y)
            effect = float(np.nanmean(y - x))

            rows.append(
                {
                    "dataset": str(dataset),
                    "classifier": str(classifier),
                    "reducer": str(reducer),
                    "n_components": (
                        None if pd.isna(n_components) else int(n_components)
                    ),
                    "metric": m,
                    "n_pairs": int(len(merged)),
                    "mean_diff_red_minus_base": effect,
                    "t_stat": t_stat,
                    "t_pvalue": t_p,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_pvalue": w_p,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(
        ["dataset", "classifier", "metric", "reducer", "n_components"]
    )

    if out_path is None:
        out_path = str(RESULTS_DIR / "stat_tests_results.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Raw input: {raw_file}")
    print(f"Wyniki testów zapisane do: {out_path}")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
