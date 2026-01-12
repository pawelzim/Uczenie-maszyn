from typing import Dict, Any, Optional
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_reducers(
    n_components: int,
    random_state: int = 42,
    kpca_kernel: str = "rbf",
    kpca_gamma: Optional[float] = None,
    n_classes: int = 2,
) -> Dict[str, Any]:

    reducers = {
        "pca": PCA(n_components=n_components, random_state=random_state),
        "ica": FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=2000,
            tol=1e-4,
            algorithm="parallel",
            whiten="unit-variance",
        ),
        "kpca": KernelPCA(
            n_components=n_components,
            kernel=kpca_kernel,
            gamma=kpca_gamma,
            fit_inverse_transform=False,
        ),
        "lda": LinearDiscriminantAnalysis(
            n_components=min(n_components, n_classes - 1)
        ),
    }

    return reducers
