import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from Perceptron import Perceptron

def linear_fn(w):
    slope = -(w[0] / w[2]) / (w[0] / w[1])
    intercept = -w[0] / w[2]

    return lambda x: slope * x + intercept

def run():
    # Generació del conjunt de mostres
    x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                               n_classes=2, n_clusters_per_class=1, class_sep=1.25,
                               random_state=0)

    y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

    perceptron = Perceptron()
    perceptron.fit(x, y)
    y_prediction = perceptron.predict(x)

    weight_fn = linear_fn(perceptron.w)
    x_plot = np.linspace(np.max(x) + 1, np.min(x) - 1)
    y_plot = weight_fn(x_plot)

    #  Resultats
    plt.figure(1)
    plt.title("Perceptron")
    plt.plot(x_plot, y_plot, color="black")
    plt.scatter(x[:, 0], x[:, 1], c=y_prediction)  # Mostram el conjunt de mostres el color indica la classe
    plt.show()

if __name__ == "__main__":
    run()