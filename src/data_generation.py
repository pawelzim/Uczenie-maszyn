from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class SyntheticConfig:
    """
    Konfiguracja generatora danych syntetycznych (sklearn.make_classification).
    """

    name: str
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int = 0
    n_repeated: int = 0
    n_classes: int = 2
    weights: Optional[Tuple[float, ...]] = None
    flip_y: float = 0.01
    class_sep: float = 1.0
    random_state: int = 42

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d["weights"] is not None:
            d["weights"] = list(d["weights"])
        return d


def generate_synthetic_dataset(
    cfg: SyntheticConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generuje (X, y) oraz zwraca metadane konfiguracji do logowania
    """
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=cfg.n_repeated,
        n_classes=cfg.n_classes,
        weights=list(cfg.weights) if cfg.weights is not None else None,
        flip_y=cfg.flip_y,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
    )
    return X, y, cfg.to_dict()


def experiment1_feature_configs(
    *,
    n_samples: int = 2000,
    feature_counts: Optional[List[int]] = None,
    informative_ratios: Optional[List[float]] = None,
    redundant_ratio: float = 0.2,
    flip_y: float = 0.01,
    class_sep: float = 1.0,
    random_state_base: int = 1000,
) -> List[SyntheticConfig]:
    """
    Konfiguracja eksperymentu 1 z roznymi liczbami cech i stosunkami cech informatywnych
    """
    if feature_counts is None:
        feature_counts = [20, 50, 100, 200]
    if informative_ratios is None:
        informative_ratios = [0.2, 0.4, 0.6, 0.8]

    configs: List[SyntheticConfig] = []
    rs = random_state_base

    for n_features in feature_counts:
        for r in informative_ratios:
            r = float(r)
            n_informative = int(round(r * n_features))
            n_informative = max(2, min(n_informative, n_features - 2))

            max_redundant = n_features - n_informative
            n_redundant = int(round(redundant_ratio * n_features))
            n_redundant = max(0, min(n_redundant, max_redundant))

            name = f"syn_E1_f{n_features}_inf{n_informative}_red{n_redundant}"

            configs.append(
                SyntheticConfig(
                    name=name,
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=n_informative,
                    n_redundant=n_redundant,
                    n_repeated=0,
                    n_classes=2,
                    weights=(0.5, 0.5),
                    flip_y=flip_y,
                    class_sep=class_sep,
                    random_state=rs,
                )
            )
            rs += 1

    return configs


def experiment2_imbalance_configs(
    *,
    n_samples: int = 3000,
    n_features: int = 100,
    n_informative: int = 20,
    imbalance_weights: List[Tuple[float, float]] = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)],
    flip_y: float = 0.01,
    class_sep: float = 1.0,
    random_state_base: int = 2000,
) -> List[SyntheticConfig]:
    """
    Konfiguracja eksperymentu 2 z roznym niezbalansowaniem klas (50/50, 70/30, 90/10)
    """
    configs: List[SyntheticConfig] = []
    rs = random_state_base

    for w0, w1 in imbalance_weights:
        name = f"syn_E2_w{int(w0*100)}_{int(w1*100)}"
        configs.append(
            SyntheticConfig(
                name=name,
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=max(0, n_features - n_informative),
                n_repeated=0,
                n_classes=2,
                weights=(w0, w1),
                flip_y=flip_y,
                class_sep=class_sep,
                random_state=rs,
            )
        )
        rs += 1

    return configs


def quick_synthetic_configs() -> List[SyntheticConfig]:
    """
    szybka konfiguracja do testow:
    1. Rozszerzona wymiarowosc do 50 cech
    2. Niezbalansowana 90:10
    """
    return [
        SyntheticConfig(
            name="syn_high_dim_50feat",
            n_samples=2000,
            n_features=50,
            n_informative=20,  # 40% informatywnych
            n_redundant=10,
            n_repeated=0,
            n_classes=2,
            weights=(0.5, 0.5),  # zbalansowane
            flip_y=0.01,
            class_sep=1.0,
            random_state=42,
        ),
        SyntheticConfig(
            name="syn_imbalanced_90_10",
            n_samples=2000,
            n_features=20,
            n_informative=8,  # 40% informatywnych
            n_redundant=4,
            n_repeated=0,
            n_classes=2,
            weights=(0.9, 0.1),  # niezbalansowane
            flip_y=0.01,
            class_sep=1.0,
            random_state=43,
        ),
    ]
