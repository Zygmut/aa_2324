import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from AdaLine import Adaline


# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=9)
y[y == 0] = -1

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Entrenam un perceptron
perceptron = Adaline(0.0005, 60)
perceptron.fit(X_transformed, y)
y_prediction = perceptron.predict(X)

#Entrenam una SVM linear (classe SVC)
svc_class = SVC(C=10000, kernel="linear")
svc_class.fit(X_transformed, y, sample_weight=None)

plt.figure(1)

#  Mostram els resultats Adaline
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.axline(xy1=(0, perceptron.intercept), slope=perceptron.slope, c="blue", label="Adaline")

#  Mostram els resultats SVM
print(svc_class.intercept_)
slope = -svc_class.coef_[0][0] / svc_class.coef_[0][1]
plt.axline(xy1=(0, -svc_class.intercept_[0] / svc_class.coef_[0][1]), slope=slope, c="cyan", label="SVM")
plt.scatter(svc_class.support_vectors_[:, 0], svc_class.support_vectors_[:, 1], facecolors="none", edgecolors="cyan")

plt.legend()
plt.show()
