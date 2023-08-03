import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm


def rounding(y_hat, threshold=0.5):
    """
    Vector Rounding
    :param y_hat: input vector
    :param threshold: threshold for rounding
    :return: binary vector
    """
    y_r = y_hat.copy()
    y_r = (y_r >= threshold)*1

    return y_r


def mse(y1, y2):
    """
    Compute a norm between two vectors
    :param y1: vector 1
    :param y2: vector 2
    :return: norm between vectors
    """
    if y1.ndim == 2:
        mse_loss = 0
        n2 = y2.shape[0]
        for i in range(y1.shape[0]):
            mse_loss += np.sum((y1[i, :]-y2[i%n2, :])**2)
        mse_loss = mse_loss**(1/2)/y1.shape[0]
        return mse_loss
    else:
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

    fig, ax = plt.subplots(figsize=(18, 6))
    ax = sns.heatmap(u.T)
    ax.set_title('Dynamic of the input vector', fontsize=16)
    ax.set_xlabel('Time t', fontsize=14)
    ax.set_ylabel('Values of vector u[t]', fontsize=14)

    print("Average number of edges in a graph is ", np.mean(np.sum(np.vstack([u[:, :], y[-1, :]]), axis=1)))


def define_periodicity(u, min_period=None, max_period=None, mode="exact", thresholds=None):
    """
    Define if the input process has periodicity
    :param max_period:
    :param mode: exact - the initial data is exact, otherwise - estimate periodicity
    :param thresholds: - thresholds for rounding
    :return:
    :param u: input data
    :return: length of period
    """
    n = u.shape[0]
    if max_period is None:
        max_period = int(n/2)

    periodicity = n
    if mode == "exact":
        for i in range(1, n + 1):
            chk = 1
            for j in range(i + 1, n):
                if not np.array_equal(u[j % i, :], u[j, :]):
                    chk = 0
                    break
            if chk == 1:
                periodicity = i+1
                break
    else:
        best_copy = np.zeros(u.shape)
        best_loss = float('inf')
        loss_arr = [mse(best_copy, u)]
        for p in tqdm(range(min_period, min(max_period+1, u.shape[0]+1))):
            u_copy = average_period(u, p)
            mse_b = float('inf')
            best_th = -1
            if thresholds is not None:
                for th in thresholds:
                    u1 = rounding(u_copy, th / 100)
                    ms = mse(u, u1)
                    if ms < mse_b:
                        best_th = th / 100
                        mse_b = ms
            #u_copy = rounding(u_copy, best_th)
            loss = mse(u, u_copy)
            loss_arr.append(loss)
            if loss < best_loss:
                best_copy = []  #u_copy.copy()
                best_loss = loss
                periodicity = p
                #print(best_loss, periodicity)
    if mode == "exact":
        return periodicity
    else:
        return periodicity, best_copy, loss_arr


def int_list(n):
    return [str(x) for x in range(n)]


def create_initial_u_y(dim, timestamps, num_ones, order):
    u = np.zeros((timestamps, dim))
    y = np.zeros((timestamps, dim))

    if num_ones is None:
        u[0, 0] = 1
    else:
        for j in range(min(num_ones, dim)):
            u[0, order[j]] = 1
    return u, y


def change_output(y, s, order):
    y = y.copy()
    if s == 1:  # add edge
        for j in range(len(y)):
            if y[order[j]] == 0:
                y[order[j]] = 1
                break
    if s == 0:  # remove edge
        for j in reversed(range(len(y))):
            if y[order[j]] == 1:
                y[order[j]] = 0
                break
    return y


def define_dim(u, getlist=None, start=None):
    n = u.shape[1]
    if start is None:
        start = 0
    list_col = []
    for i in range(start, u.shape[1]):
        s = sum(u[:, i])
        if s == 0:  # or (s == u.shape[0] and s != 1):
            list_col.append(i)
            n = n - 1
        elif i > 0:
            r1 = np.linalg.matrix_rank(u[:, :i])
            r2 = np.linalg.matrix_rank(u[:, :i+1])
            if r1 == r2:
                list_col.append(i)

    n = u.shape[1] - len(list_col)

    if getlist:
        return n, list_col
    else:
        return n


def get_subarray(u, list_col):
    mask = np.ones(u.shape[1], dtype=bool)
    mask[list_col] = False
    return u[:, mask].copy()


def compute_average(dic, ind):
    out = dict()
    for i in dic:
        min_dif = 100000000
        max_dif = -10000000
        out[i] = dict()
        for j in dic[i]:
            out[i][j] = 0
            for k in range(len(dic[i][j])):
                if dic[i][j][k][ind] is not None:
                    out[i][j] += dic[i][j][k][ind]
                if dic[i][j][k][1] is not None:
                    v = dic[i][j][k][2] - dic[i][j][k][3] - dic[i][j][k][4]
                    if v > max_dif:
                        max_dif = v
                    if v < min_dif:
                        min_dif = v
            out[i][j] = out[i][j] / len(dic[i][j])
        print("Nodes = ", i)
        print("Min/Max difference:", min_dif, max_dif)
    return out


def average_period(u, period):
    n = u.shape[0]
    full_row = n // period
    av_arr = np.zeros((period, u.shape[1]))

    for i in range(full_row):
        av_arr[:, :] += u[period*i:period*(i+1), :]
    for i in range(full_row*period, n):
        av_arr[i % period, :] += u[i, :]

    #av_arr[:period, :] += u[:period, :]#.copy()
    #for i in range(period, n):
    #    av_arr[i % period, :] += u[i, :]
    av_arr[:(n - full_row * period), :] /= (full_row + 1)
    av_arr[(n - full_row * period):, :] /= full_row
    #for i in range(period):
    #    if n - full_row * period > i:
    #        av_arr[i, :] /= (full_row + 1)
    #    else:
    #        av_arr[i, :] /= full_row
    return av_arr


def prepare_input_data(u, y):
    inp_dim, indices = define_dim(u, getlist=True)
    return get_subarray(u, indices), get_subarray(y, indices)
