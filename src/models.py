from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier


def get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    return {
        "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            solver="adam",
            max_iter=800,
            tol=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            alpha=1e-4,
            random_state=random_state,
        ),
        # "logreg": LogisticRegression(max_iter=2000, solver="lbfgs"),
        # "rf": RandomForestClassifier(
        #     n_estimators=300, random_state=random_state, n_jobs=-1
        # ),
    }
