from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class CCPCA:
    n_components: int
    random_state: int = 42

    # Pola ustawiane w fit()
    pca_: Optional[PCA] = None
    centroids_: Optional[np.ndarray] = None
    classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CCPCA":
        X = np.asarray(X)
        y = np.asarray(y)

        # Walidacja n_components
        n_features = X.shape[1]
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed n_features ({n_features})"
            )

        self.classes_ = np.unique(y)

        # Centroidy: indeksowane w self.classes_
        centroids = np.zeros((len(self.classes_), n_features), dtype=float)
        for i, cls in enumerate(self.classes_):
            centroids[i] = X[y == cls].mean(axis=0)

        self.centroids_ = centroids

        # Centrowanie klasowe danych treningowych
        Xc = self._center_by_class_centroid(X, y)

        # PCA po CCPCA centrowaniu
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca_.fit(Xc)
        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.pca_ is None or self.centroids_ is None or self.classes_ is None:
            raise RuntimeError("CCPCA must be fitted before calling transform().")

        X = np.asarray(X)
        y = np.asarray(y)

        Xc = self._center_by_class_centroid(X, y)
        return self.pca_.transform(Xc)

    def _center_by_class_centroid(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # mapowanie klasy -> indeks w tablicy centroidow
        class_to_index = {cls: i for i, cls in enumerate(self.classes_)}

        # Sprawdzenie, czy wszystkie klasy w y są znane z treningu
        unknown_classes = set(y) - set(self.classes_)
        if unknown_classes:
            raise ValueError(
                f"Unknown classes in transform/predict: {unknown_classes}. "
                f"Allowed classes: {set(self.classes_)}"
            )

        idx = np.array([class_to_index[cls] for cls in y])

        Xc = X - self.centroids_[idx]
        return Xc

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Odwrotna transformacja PCA.
        """
        if self.pca_ is None:
            raise RuntimeError(
                "CCPCA must be fitted before calling inverse_transform()."
            )

        return self.pca_.inverse_transform(X_reduced)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_components": self.n_components,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "CCPCA":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
