from dataclasses import dataclass
from typing import Optional, List

WEIGHT_NOT_DEFINED_ERR = "There are no weights defined. Have you called `fit` with your data?"

@dataclass
class Perceptron:
    eta: float = 0.0001
    threshold: float = 10.00
    epochs: int = 10
    __costs: Optional[List[float]] = None
    __w: Optional[List[float]] = None

    def __post_init__(self):
        if not (0.0 <= self.eta <= 1.0):
            raise ValueError("eta should be a float between 0.0 and 1.0")


    @property
    def slope(self):
        """Slope of the linear function"""
        if self.__w != None:
            return -(self.__w[0] / self.__w[2]) / (self.__w[0] / self.__w[1])
        else:
            raise ValueError(WEIGHT_NOT_DEFINED_ERR)

    @property
    def intercept(self):
        """Intercept of the linear function"""
        if self.__w != None:
            return -self.__w[0] / self.__w[2]
        else:
            raise ValueError(WEIGHT_NOT_DEFINED_ERR)

    @property
    def weights(self):
        if self.__w != None:
            return self.__w
        else:
            raise ValueError(WEIGHT_NOT_DEFINED_ERR)

    @property
    def costs(self):
        if self.__costs != None:
            return self.__costs
        else :
            raise ValueError("There are no costs defined. Have you called `fit` with your data?")

    def fit(self, data, results) -> None:
        self.__costs = []
        self.__w = [0] * (len(data[0]) + 1)

        while len(self.__costs) <= self.epochs or self.__costs[-1] <= self.threshold:
            error = False
            error_acc = 0
            for x_hat, y_hat in zip(data, results):
                delta_error = y_hat - self.predict(x_hat)
                error_acc += delta_error**2

                delta_w = self.eta * delta_error
                self.__w[0] += delta_w

                for i in range(len(x_hat)):
                    self.__w[1 + i] += delta_w * x_hat[i]

                if delta_error != 0.0 and not error:
                    error = True

            self.__costs.append(error_acc / 2)

    def __net_output(self, data) -> float:
        acc = self.__w[0]

        for val, w in zip(data, self.__w[1:]):
            acc += val * w

        return acc

    def predict(self, data) -> float:
        return 1 if self.__net_output(data) >= 0 else -1
