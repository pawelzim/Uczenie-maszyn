from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(50, 50),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        ),
        "logreg": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "rf": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
    }
