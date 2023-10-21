from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

USE_MIN_MAX_SCALE = True
RANDOM_STATE = 27

def easy_pred(use_kernel, data, truth, test_data, use_random_state=RANDOM_STATE):
    svc = SVC(kernel=use_kernel, random_state=use_random_state)
    svc.fit(data, truth)
    return svc.predict(test_data)

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = MinMaxScaler() if USE_MIN_MAX_SCALE else StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

easy_precision_scores = lambda x: list(map(lambda x: precision_score(y_test, x), map(lambda x: easy_pred(x, X_transformed, y_train, X_test_transformed), x)))

compare = [
    ["linear", lambda x, y: x.dot(y.T)],
    ["rbf", lambda x, y, gamma=10: np.exp(-gamma * distance_matrix(x, y) ** 2)],
    ["poly", lambda x, y, gamma=10, r=0, d=2: (gamma * x.dot(y.T) + r) ** d]
]

print(list(map(easy_precision_scores, compare)))