
from dataclasses import dataclass
import numpy as np

@dataclass
class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epoch : int
        Passes over the training dataset.
    """

    eta: float = 0.01
    epoch: int = 10
    __cost = None
    __w = None

    def __post_init__(self):
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta value must be between 0.0 and 1.0")

    @property
    def slope(self):
        """Slope of the linear function"""
        return -(self.__w[0] / self.__w[2]) / (self.__w[0] / self.__w[1])

    @property
    def intercept(self):
        """Intercept of the linear function"""
        return -self.__w[0] / self.__w[2]

    @property
    def cost(self):
        return self.__cost

    def fit(self, x, y):
        """Fit training data.

        Parameters
        ----------
        data :  {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
        y :     array-like, shape = [n_samples]
                Target values.

        """

        self.__w = np.zeros(1 + x.shape[1])
        self.__cost = []

        for _ in range(self.epoch):

            epoch_err = 0
            for x_it, y_it in zip(x, y):
                error = y_it - self.predict(x_it)
                epoch_err += error ** 2

                delta_w = self.eta * error
                self.__w[0] += delta_w
                self.__w[1:] += delta_w * x_it

            self.__cost.append(epoch_err / 2)

    def predict(self, x):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        res = np.dot(x, self.__w[1:]) + self.__w[0]

        return np.where(res >= 0, 1, -1)
