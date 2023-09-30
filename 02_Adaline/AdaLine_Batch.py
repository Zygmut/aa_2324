import numpy as np
from dataclasses import dataclass

@dataclass
class Adaline:
    """ADAptive LInear NEuron classifier.
       Gradient Descent

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epoch: int
        Passes over the training dataset.
    """

    eta: float = 0.01
    epoch: int = 50
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
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.__w = np.zeros(1 + x.shape[1])
        self.__cost = []

        for _ in range(self.epoch):
            errors = (y - self.predict(x))
            self.__w[0] += self.eta * errors.sum()
            self.__w[1:] += self.eta * x.T.dot(errors)

            self.__cost.append((errors**2).sum() / 2)

    def predict(self, x):
        """Calculate the prediction"""
        return np.dot(x, self.__w[1:]) + self.__w[0]