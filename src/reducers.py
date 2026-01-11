from typing import Dict, Any, Optional
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ccpca import CCPCA


def get_reducers(
    n_components: int,
    random_state: int = 42,
    kpca_kernel: str = "rbf",
    kpca_gamma: Optional[float] = None,
) -> Dict[str, Any]:

    reducers = {
        "pca": PCA(n_components=n_components, random_state=random_state),
        "ica": FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=2000,
        ),
        "kpca": KernelPCA(
            n_components=n_components,
            kernel=kpca_kernel,
            gamma=kpca_gamma,
            fit_inverse_transform=False,
        ),
        "lda": LinearDiscriminantAnalysis(n_components=1),
        "ccpca": CCPCA(
            n_components=n_components,
            random_state=random_state,
        ),
    }

    return reducers
