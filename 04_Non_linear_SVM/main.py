from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

USE_MIN_MAX_SCALE = True
RANDOM_STATE =33 


def easy_pred(use_kernel, data, truth, test_data, use_random_state=RANDOM_STATE):
    svc = SVC(kernel=use_kernel, random_state=use_random_state)
    svc.fit(data, truth)
    return svc.predict(test_data)


def linear_kernel(x1, x2):
    return x1.dot(x2.T)


def gauss_kernel(x1, x2, sigma=1/2):
    gamma = -1 / (2 * sigma ** 2)
    return np.exp(gamma * distance_matrix(x1, x2) ** 2)  #


def poly_kernel(x1, x2, degree=3, sigma=1/2):
    gamma = 1 / (2 * sigma ** 2)
    return (gamma * x1.dot(x2.T)) ** degree


X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

train_data, test_data, train_pred, test_pred = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = MinMaxScaler() if USE_MIN_MAX_SCALE else StandardScaler()
train_trans = scaler.fit_transform(train_data)
test_tran = scaler.transform(test_data)

compare_list = [
    ["linear", linear_kernel],
    ["rbf", gauss_kernel],
    ["poly", poly_kernel]
]

base_precision_scores = []

for comparison in compare_list:
    predictions = [easy_pred(kernel, train_trans, train_pred, test_tran)
                   for kernel in comparison]

    base_precision_scores.append([precision_score(test_pred, prediction)
                                  for prediction in predictions])

print(base_precision_scores)

poly = PolynomialFeatures(3)
poly_train_tran = poly.fit_transform(train_data)
poly_test_tran = poly.transform(test_data)


poly_predictions = [easy_pred(kernel, poly_train_tran, train_pred, poly_test_tran)
               for kernel in compare_list[0]]

poly_precision_scores = [precision_score(test_pred, prediction)
                              for prediction in predictions]

print(poly_precision_scores)
