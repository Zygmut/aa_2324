import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from Perceptron import Perceptron


def run():
    # Generació del conjunt de mostres
    x, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.25,
        random_state=0,
    )

    y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.

    x = x.tolist()
    y = y.tolist()

    perceptron = Perceptron(0.00000005)
    perceptron.fit(x, y)

    y_prediction = list(map(perceptron.predict, x))

    #  Resultats
    plt.figure(1)
    plt.title("Perceptron")
    plt.axline(xy1=(0, perceptron.intercept), slope=perceptron.slope)
    plt.scatter(
        [val[0] for val in x], [val[1] for val in x], c=y_prediction
    )  # Mostram el conjunt de mostres el color indica la classe

    plt.figure(2)
    plt.plot(perceptron.costs, marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Sum of squared error")
    plt.show()


if __name__ == "__main__":
    run()
