import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Auto-detekcja katalogów
# =========================
def resolve_tables_dir() -> Path:
    for p in [Path("tables"), Path("results") / "tables"]:
        if p.exists() and p.is_dir():
            return p
    return Path("tables")


def resolve_figures_dir() -> Path:
    for p in [Path("figures"), Path("results") / "figures"]:
        if p.exists() and p.is_dir():
            return p
    return Path("figures")


TABLES_DIR = resolve_tables_dir()
FIGURES_DIR = resolve_figures_dir()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Konfiguracja E1
# =========================
E1_REDUCERS = ["pca", "ica", "kpca", "lda"]  # pomijamy ccpca
METRIC_MEAN = "bac_mean"
METRIC_STD = "bac_std"

# Stałe do wykresów E1 (wybierz sensowne przekroje)
FOCUS_N_COMPONENTS = 10
FOCUS_INF_PCT = 40
FOCUS_N_FEATURES = 100


# =========================
# Parsowanie nazw datasetów
# syn_E1_f100_inf40
# =========================
def parse_e1_dataset(name: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(name, str):
        return None, None
    mf = re.search(r"_f(\d+)", name)
    mi = re.search(r"_inf(\d+)", name)
    n_features = int(mf.group(1)) if mf else None
    inf_pct = int(mi.group(1)) if mi else None
    return n_features, inf_pct


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")
    return pd.read_csv(path)


def ensure_reducer_column(df: pd.DataFrame, fallback_reducer: str) -> pd.DataFrame:
    if "reducer" in df.columns:
        return df
    df = df.copy()
    df["reducer"] = fallback_reducer
    return df


def load_e1() -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    # Wczytujemy wszystkie pliki E1; w nich powinno być też "none"
    for r in E1_REDUCERS:
        p = TABLES_DIR / f"reduction_cv_E1_{r}.csv"
        if not p.exists():
            continue
        df = read_csv(p)
        df = ensure_reducer_column(df, r)
        dfs.append(df)

    if not dfs:
        raise RuntimeError(
            f"Nie znaleziono plików reduction_cv_E1_*.csv w {TABLES_DIR.resolve()}"
        )

    e1 = pd.concat(dfs, ignore_index=True)
    e1 = e1[e1["dataset"].astype(str).str.contains("syn_E1", na=False)].copy()

    e1["n_features"], e1["inf_pct"] = zip(*e1["dataset"].map(parse_e1_dataset))
    e1 = e1.dropna(subset=["n_features", "inf_pct"])
    e1["n_features"] = e1["n_features"].astype(int)
    e1["inf_pct"] = e1["inf_pct"].astype(int)

    if "n_components" in e1.columns:
        e1["n_components"] = e1["n_components"].astype(int)

    # Walidacja kolumn metryki
    if METRIC_MEAN not in e1.columns:
        raise RuntimeError(
            f"Brak kolumny {METRIC_MEAN} w danych E1. Dostępne: {list(e1.columns)}"
        )

    return e1


def save_plot(stem: str) -> None:
    out = FIGURES_DIR / f"{stem}.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Zapisano: {out.name}")


def reducer_label(r: str) -> str:
    return "brak redukcji" if r == "none" else r.upper()


# =========================
# Wykresy E1 bez heatmap
# =========================
def plot_e1_bac_vs_n_features(
    e1: pd.DataFrame, classifier: str, inf_pct: int, n_components: int
) -> None:
    """
    Punktowo/liniowo: BAC vs liczba cech
    Stałe: inf_pct, n_components
    Linie: none, PCA, ICA, KPCA, LDA
    """
    df = e1[
        (e1["classifier"].astype(str) == classifier)
        & (e1["inf_pct"] == inf_pct)
        & (e1["n_components"] == n_components)
    ].copy()

    if df.empty:
        print(
            f"[INFO] Brak danych dla wykresu BAC vs n_features: clf={classifier}, inf={inf_pct}, nc={n_components}"
        )
        return

    agg = (
        df.groupby(["n_features", "reducer"], as_index=False)[METRIC_MEAN]
        .mean()
        .sort_values(["reducer", "n_features"])
    )

    plt.figure(figsize=(10, 6))
    for r, g in agg.groupby("reducer"):
        g = g.sort_values("n_features")
        plt.plot(
            g["n_features"],
            g[METRIC_MEAN],
            marker="o",
            linewidth=2,
            label=reducer_label(r),
        )

    plt.title(
        f"E1: BAC vs liczba cech | klasyfikator: {classifier} | inf={inf_pct}% | n_components={n_components}"
    )
    plt.xlabel("Liczba cech (f)")
    plt.ylabel("BAC (Balanced Accuracy)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(ncol=2)
    save_plot(f"E1_BAC_vs_nfeatures_{classifier}_inf{inf_pct}_nc{n_components}")


def plot_e1_bac_vs_inf_pct(
    e1: pd.DataFrame, classifier: str, n_features: int, n_components: int
) -> None:
    """
    Punktowo/liniowo: BAC vs % informatywnych
    Stałe: n_features, n_components
    Linie: none, PCA, ICA, KPCA, LDA
    """
    df = e1[
        (e1["classifier"].astype(str) == classifier)
        & (e1["n_features"] == n_features)
        & (e1["n_components"] == n_components)
    ].copy()

    if df.empty:
        print(
            f"[INFO] Brak danych dla wykresu BAC vs inf_pct: clf={classifier}, f={n_features}, nc={n_components}"
        )
        return

    agg = (
        df.groupby(["inf_pct", "reducer"], as_index=False)[METRIC_MEAN]
        .mean()
        .sort_values(["reducer", "inf_pct"])
    )

    plt.figure(figsize=(10, 6))
    for r, g in agg.groupby("reducer"):
        g = g.sort_values("inf_pct")
        plt.plot(
            g["inf_pct"],
            g[METRIC_MEAN],
            marker="o",
            linewidth=2,
            label=reducer_label(r),
        )

    plt.title(
        f"E1: BAC vs % cech informatywnych | klasyfikator: {classifier} | f={n_features} | n_components={n_components}"
    )
    plt.xlabel("Udział cech informatywnych [%] (inf)")
    plt.ylabel("BAC (Balanced Accuracy)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(ncol=2)
    plt.xticks(sorted(agg["inf_pct"].unique()))
    save_plot(f"E1_BAC_vs_infpct_{classifier}_f{n_features}_nc{n_components}")


def plot_e1_bar_methods_for_fixed_config(
    e1: pd.DataFrame, classifier: str, n_features: int, inf_pct: int, n_components: int
) -> None:
    """
    Słupkowy: porównanie metod dla jednej konfiguracji (f, inf, nc)
    Słupki: none + reduktory
    Błędy: bac_std (jeżeli dostępne)
    """
    df = e1[
        (e1["classifier"].astype(str) == classifier)
        & (e1["n_features"] == n_features)
        & (e1["inf_pct"] == inf_pct)
        & (e1["n_components"] == n_components)
    ].copy()

    if df.empty:
        print(
            f"[INFO] Brak danych dla słupków: clf={classifier}, f={n_features}, inf={inf_pct}, nc={n_components}"
        )
        return

    agg = df.groupby("reducer", as_index=False).agg(
        mean=(METRIC_MEAN, "mean"),
        std=(METRIC_STD, "mean") if METRIC_STD in df.columns else (METRIC_MEAN, "std"),
    )

    order = ["none"] + [
        r for r in ["pca", "ica", "kpca", "lda"] if r in set(agg["reducer"])
    ]
    agg["reducer"] = pd.Categorical(agg["reducer"], categories=order, ordered=True)
    agg = agg.sort_values("reducer")

    x = [reducer_label(r) for r in agg["reducer"].astype(str)]
    y = agg["mean"].astype(float).tolist()
    yerr = agg["std"].astype(float).tolist() if "std" in agg.columns else None

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, yerr=yerr, capsize=6)

    plt.title(
        f"E1: Porównanie metod | {classifier} | f={n_features}, inf={inf_pct}%, n_components={n_components}"
    )
    plt.xlabel("Metoda")
    plt.ylabel("BAC (Balanced Accuracy)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.6)

    # wartości na słupkach
    for i, v in enumerate(y):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    save_plot(
        f"E1_bar_methods_{classifier}_f{n_features}_inf{inf_pct}_nc{n_components}"
    )


def plot_e1_delta_vs_n_features(
    e1: pd.DataFrame, classifier: str, inf_pct: int, n_components: int
) -> None:
    """
    Punktowo/liniowo: ΔBAC vs liczba cech
    ΔBAC = BAC(reducer) - BAC(none)
    Stałe: inf_pct, n_components
    Linie: tylko reduktory (bez none)
    """
    df = e1[
        (e1["classifier"].astype(str) == classifier)
        & (e1["inf_pct"] == inf_pct)
        & (e1["n_components"] == n_components)
    ].copy()

    if df.empty:
        return

    none = (
        df[df["reducer"].astype(str) == "none"]
        .groupby("n_features", as_index=False)[METRIC_MEAN]
        .mean()
    )
    if none.empty:
        return
    none = none.rename(columns={METRIC_MEAN: "none_mean"})

    red = (
        df[df["reducer"].astype(str) != "none"]
        .groupby(["n_features", "reducer"], as_index=False)[METRIC_MEAN]
        .mean()
    )
    merged = red.merge(none, on="n_features", how="left")
    merged["delta"] = merged[METRIC_MEAN] - merged["none_mean"]

    plt.figure(figsize=(10, 6))
    for r, g in merged.groupby("reducer"):
        g = g.sort_values("n_features")
        plt.plot(g["n_features"], g["delta"], marker="o", linewidth=2, label=r.upper())

    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.title(
        f"E1: ΔBAC vs liczba cech | {classifier} | inf={inf_pct}% | n_components={n_components}"
    )
    plt.xlabel("Liczba cech (f)")
    plt.ylabel("ΔBAC = BAC(redukcja) − BAC(brak redukcji)")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(ncol=2)
    save_plot(f"E1_deltaBAC_vs_nfeatures_{classifier}_inf{inf_pct}_nc{n_components}")


def plot_e1_delta_vs_inf_pct(
    e1: pd.DataFrame, classifier: str, n_features: int, n_components: int
) -> None:
    """
    Punktowo/liniowo: ΔBAC vs % informatywnych
    ΔBAC = BAC(reducer) - BAC(none)
    Stałe: n_features, n_components
    Linie: tylko reduktory
    """
    df = e1[
        (e1["classifier"].astype(str) == classifier)
        & (e1["n_features"] == n_features)
        & (e1["n_components"] == n_components)
    ].copy()

    if df.empty:
        return

    none = (
        df[df["reducer"].astype(str) == "none"]
        .groupby("inf_pct", as_index=False)[METRIC_MEAN]
        .mean()
    )
    if none.empty:
        return
    none = none.rename(columns={METRIC_MEAN: "none_mean"})

    red = (
        df[df["reducer"].astype(str) != "none"]
        .groupby(["inf_pct", "reducer"], as_index=False)[METRIC_MEAN]
        .mean()
    )
    merged = red.merge(none, on="inf_pct", how="left")
    merged["delta"] = merged[METRIC_MEAN] - merged["none_mean"]

    plt.figure(figsize=(10, 6))
    for r, g in merged.groupby("reducer"):
        g = g.sort_values("inf_pct")
        plt.plot(g["inf_pct"], g["delta"], marker="o", linewidth=2, label=r.upper())

    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.title(
        f"E1: ΔBAC vs % informatywnych | {classifier} | f={n_features} | n_components={n_components}"
    )
    plt.xlabel("Udział cech informatywnych [%] (inf)")
    plt.ylabel("ΔBAC = BAC(redukcja) − BAC(brak redukcji)")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(ncol=2)
    plt.xticks(sorted(merged["inf_pct"].unique()))
    save_plot(f"E1_deltaBAC_vs_infpct_{classifier}_f{n_features}_nc{n_components}")


def main():
    print(f"[INFO] tables:  {TABLES_DIR.resolve()}")
    print(f"[INFO] figures: {FIGURES_DIR.resolve()}")

    e1 = load_e1()
    classifiers = sorted(e1["classifier"].astype(str).unique())
    ncs = sorted(e1["n_components"].unique()) if "n_components" in e1.columns else []

    print(f"[INFO] Klasyfikatory: {classifiers}")
    print(f"[INFO] n_components: {ncs}")

    for clf in classifiers:
        plot_e1_bac_vs_n_features(
            e1, classifier=clf, inf_pct=FOCUS_INF_PCT, n_components=FOCUS_N_COMPONENTS
        )
        plot_e1_bac_vs_inf_pct(
            e1,
            classifier=clf,
            n_features=FOCUS_N_FEATURES,
            n_components=FOCUS_N_COMPONENTS,
        )

        plot_e1_bar_methods_for_fixed_config(
            e1,
            classifier=clf,
            n_features=FOCUS_N_FEATURES,
            inf_pct=FOCUS_INF_PCT,
            n_components=FOCUS_N_COMPONENTS,
        )
        plot_e1_delta_vs_n_features(
            e1, classifier=clf, inf_pct=FOCUS_INF_PCT, n_components=FOCUS_N_COMPONENTS
        )
        plot_e1_delta_vs_inf_pct(
            e1,
            classifier=clf,
            n_features=FOCUS_N_FEATURES,
            n_components=FOCUS_N_COMPONENTS,
        )

    print("[INFO] Gotowe.")


if __name__ == "__main__":
    main()
