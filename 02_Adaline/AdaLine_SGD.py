import numpy as np
from numpy.random import seed
from dataclasses import dataclass
from typing import Optional

@dataclass
class Adaline(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)
    epoch : int (default: 50)
        Passes over the training dataset.
    shuffle : bool (default: True)
        Shuffles training dat every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
    """

    eta: float = 0.01
    epoch: int = 50
    shuffle: bool = True
    random_state: Optional[int] = None
    __cost = None
    __w = None

    def __post_init__(self):
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta value must be between 0.0 and 1.0")

        if self.random_state:
            seed(self.random_state)

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
        x : {array-like}, shape = [n_samples, n_features]
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

            if self.shuffle:
                x, y = self.__shuffle(x, y)

            epoch_err = 0

            for xi, target in zip(x, y):
                error = (target - self.predict(xi))
                epoch_err += error ** 2

                delta_w = self.eta * error
                self.__w[0] += delta_w
                self.__w[1:] += delta_w * xi

            self.__cost.append(epoch_err / 2)

    def __shuffle(self, x, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def predict(self, x):
        res = self.__w.T
        return np.dot(x, res[1:]) + res[0]
