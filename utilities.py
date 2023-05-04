import numpy as np
import pandas as pd
import math


def rounding(y_hat, threshold=0.5):
    """
    Vector Rounding
    :param y_hat: input vector
    :param threshold: threshold for rounding
    :return: binary vector
    """
    for i in range(y_hat.shape[0]):
        for j in range(y_hat.shape[1]):
            y_hat[i, j] = 1 if y_hat[i, j] >= threshold else 0
    return y_hat


def mse(y1, y2):
    """
    Compute a norm between two vectors
    :param y1: vector 1
    :param y2: vector 2
    :return: norm between vectors
    """
    return np.linalg.norm(y1-y2)


def cum_mse(y1, y2):
    """
    Compute a cumulative norm between two vectors
    :param y1: vector 1
    :param y2: vector 2
    :return: cumulative norm between vectors
    """
    err = np.zeros(y1.shape[0])
    err[0] = mse(y1[0, :], y2[0, :])

    for i in range(y1.shape[0]):
        err[i] = mse(y1[i, :], y2[i, :])+err[i-1]
    return err


def describe_data(u, y, figure_size=None):
    """
    Describe the Initial Data
    :param u: input data
    :param y: output data
    :param figure_size: plot size of figures
    """

    if figure_size is None:
        figure_size = (12, 3)
    pd.DataFrame(np.sum(np.vstack([u[:, :], y[-1, :]]), axis=1)).plot(title='Number of edges in a graph',
                                                                      figsize=figure_size, legend=False)
    print("Average number of edges in a graph is ", np.mean(np.sum(np.vstack([u[:, :], y[-1, :]]), axis=1)))
