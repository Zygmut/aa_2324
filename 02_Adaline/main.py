import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
# Com les classes es diuen igual basta canviar el fitxer ;)
from AdaLine_SGD import Adaline

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=9)

y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

perceptron = Adaline()
perceptron.fit(X_std, y)

y_prediction = perceptron.predict(X)

#  Mostram els resultats
plt.figure(1)
plt.scatter(X_std[:, 0], X_std[:, 1], c=y)

# Dibuixem la recta de separació
plt.axline(xy1=(0, perceptron.intercept), slope=perceptron.slope)

plt.figure(2)
plt.plot(perceptron.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum of squared error')
plt.show()
