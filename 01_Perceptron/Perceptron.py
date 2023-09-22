
from dataclasses import dataclass
import numpy as np

@dataclass
class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    eta: float = 0.01
    n_iter: int = 10
    w: np.ndarray[np.float64] = None


    def fit(self, data, y):

        """Fit training data.

        Parameters
        ----------
        Xdata : {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
        y :     array-like, shape = [n_samples]
                Target values.

        """

        self.w = np.zeros(1 + data.shape[1])  # First position corresponds to threshold

        for _ in range(self.n_iter):

            for x_it, y_it in zip(data, y):
                delta_w = self.eta * (y_it - self.predict(x_it))

                self.w[0] += delta_w
                self.w[1:] += delta_w * x_it

    def predict(self, x):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        res = np.dot(x, self.w[1:]) + self.w[0]

        return np.where(res >= 0, 1, -1)
