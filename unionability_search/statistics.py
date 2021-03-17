import numpy as np
from sklearn import datasets
from scipy.stats import f

EPSILON = 10e-10  # only to prevent division by zero


def mean_vector_similarity(X, Y):
    x_mean = np.mean(X, axis=0)
    y_mean = np.mean(Y, axis=0)
    sim = float((x_mean.dot(y_mean)) / (np.linalg.norm(x_mean)
                                        * np.linalg.norm(y_mean) + EPSILON))
    return sim
