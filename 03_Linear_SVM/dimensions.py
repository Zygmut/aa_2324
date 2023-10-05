import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)

# Estandaritzar les dades: StandardScaler
scaler = StandardScaler()
X_tr_trans = scaler.fit_transform(X_train)
X_ts_trans = scaler.transform(X_test)

# Entrenam una SVM linear (classe SVC)
svc_class = SVC(C=10000, kernel="linear")
svc_class.fit(X_tr_trans, y_train, sample_weight=None)

# Prediccio
predict = svc_class.predict(X_ts_trans)

# Metrica

print(np.count_nonzero(y_test == predict) / y_test.size)

