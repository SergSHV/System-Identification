import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    if y1.ndim == 2:
        mse_loss = 0
        for i in range(y1.shape[0]):
            mse_loss += np.linalg.norm(y1[i, :]-y2[i, :])
        mse_loss = mse_loss/y1.shape[0]
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
    ax.set_title('Dynamic of the input vector')
    ax.set_xlabel('Time t', fontsize=14)
    ax.set_ylabel('Values of vector u[t]', fontsize=14)

    print("Average number of edges in a graph is ", np.mean(np.sum(np.vstack([u[:, :], y[-1, :]]), axis=1)))


def define_periodicity(u):
    """
    Define if the input process has periodicity
    :param u: input data
    :return: length of period
    """
    n = u.shape[0]
    periodicity = n
    for i in range(1, n+1):
        u_copy = np.tile(u[:i, :], (n//i + 1, 1))[:n, :]
        if np.array_equal(u, u_copy):
            periodicity = i
            break
    return periodicity


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


def graph_simple_sequence(n, timestamps, num_ones=None, s=None, order="fixed"):
    """

    :param n: number of nodes
    :param timestamps: period length
    :param num_ones: number of 1 in the initial graph G0
    :param s: type of action (0 - remove edge, 1 - add edge)
    :param order: order of edges (fixed, random)
    :return:
    """
    dim = int(n * (n - 1) / 2)
    order_inc, order_dec = generate_edge_order(dim, order)
    u, y = create_initial_u_y(dim, timestamps, num_ones, order_dec if s == 0 else order_inc)

    if s is None:
        s = 1

    for t in range(0, timestamps):
        y[t, :] = u[t, :].copy()

        if sum(y[t, :]) == 0:
            s = 1
        elif sum(u[t, :]) == len(u[t, :]):
            s = 0

        y[t, :] = change_output(y[t, :], s, order_dec if s == 0 else order_inc).copy()

        if t != timestamps - 1:
            u[t + 1, :] = y[t, :].copy()

    return u, y


def graph_simple_sequence2(n, timestamps, num_ones=None, s=None, constant=[6, 3, 2], order="fixed"):
    """
    :param n: number of nodes
    :param timestamps: period length
    :param num_ones: number of 1 in the initial graph G0
    :param s: type of action (0 - remove edge, 1 - add edge)
    :param constant: number of periods to keep the graph
    :return:
    """
    dim = int(n * (n - 1) / 2)
    order_inc, order_dec = generate_edge_order(dim, order)
    u, y = create_initial_u_y(dim, timestamps, num_ones, order_dec if s == 0 else order_inc)

    if s is None:
        s = 1

    c = 0
    for t in range(0, timestamps):
        y[t, :] = u[t, :].copy()

        if sum(y[t, :]) == 0:
            if c > constant[0]:
                c, s = 0, 1
            else:
                c += 1
        elif sum(u[t, :]) == len(u[t, :]):
            if c > constant[2]:
                c, s = 0, 0
            else:
                c += 1
        elif sum(u[t, :]) == int(len(u[t, :]) / 2):
            c += 1
            if c > constant[1] + 1:
                c = 0
            else:
                c += 1

        if c == 0:
            y[t, :] = change_output(y[t, :], s, order_dec if s == 0 else order_inc).copy()

        if t != timestamps - 1:
            u[t + 1, :] = y[t, :].copy()

    return u, y


def generate_edge_order(dim, order):
    order_inc = list(range(dim))
    order_dec = list(range(dim))
    if order == "random":
        order_inc = np.random.permutation(dim)
        order_dec = np.random.permutation(dim)
    return order_inc, order_dec


def generate_g(n, num_periods=20, per_length=None, changes=1):
    period_length = n*(n-1) if per_length is None else per_length
    dim = int(n * (n - 1) / 2)

    order_inc, order_dec = generate_edge_order(dim, order="fixed")
    u, y = create_initial_u_y(dim, timestamps=num_periods*period_length, num_ones=int(dim/2), order=order_inc)

    for t in range(period_length):
        y[t, :] = u[t, :].copy()

        for ch in range(changes):
            curr_dist = np.count_nonzero(u[t, :]!=u[0, :])
            remained_changes = (period_length-t-1)*changes + (changes - ch)

            if curr_dist < remained_changes:
                p_add = 1 if sum(u[t, :]) != len(u[t, :]) else 0
                p_remove = 1 if sum(u[t, :]) != 0 else 0
                p_keep = (p_add+p_remove)/4 if (p_add+p_remove)!=0 else 1
                p_sum = p_add + p_remove + p_keep
                action = np.random.choice(3, p=[p_remove/p_sum, p_keep/p_sum, p_add/p_sum])  # 0=remove,1=keep,2=add
                if action == 0:  # remove edge
                    l_edges = []
                    for l in range(u.shape[1]):
                        if u[t, l]==1:
                            l_edges.append(l)
                    edge = np.random.choice(l_edges)
                    y[t,edge]=0

                if action == 2: # add edge
                    l_edges = []
                    for l in range(u.shape[1]):
                        if u[t,l]==0:
                            l_edges.append(l)
                    edge = np.random.choice(l_edges)
                    y[t,edge]=1
            else:
                l_add = []
                l_remove = []
                l_keep = []
                for l in range(u.shape[1]):
                    a1 = np.concatenate([u[0, :l], u[0, l+1:]])
                    a2 = np.concatenate([u[t, :l], u[t, l+1:]])
                    new_d = np.count_nonzero(a1!=a2)
                    if curr_dist == new_d and curr_dist == remained_changes:
                        l_keep.append(l)
                    if curr_dist != new_d:
                        if u[0, l] == 1:
                            l_add.append(l)
                        if u[0, l] == 0:
                            l_remove.append(l)
                p_add = 1 if sum(u[t, :]) != len(u[t, :]) and len(l_add) > 0 else 0
                p_remove = 1 if sum(u[t, :]) != 0 and len(l_remove) > 0 else 0

                p_keep = 0
                if len(l_keep)>0:
                    p_keep = (p_add+p_remove)/4
                    if p_keep == 0:
                        p_keep = 1
                p_sum = p_add + p_remove + p_keep
                action = np.random.choice(3, p=[p_remove/p_sum, p_keep/p_sum, p_add/p_sum])  # 0=remove,1=keep,2=add
                if action == 0:  # remove edge
                    edge = np.random.choice(l_remove)
                    y[t,edge]=0

                if action == 2:  # add edge
                    edge = np.random.choice(l_add)
                    y[t,edge]=1

        if t != period_length-1:
            u[t + 1, :] = y[t, :].copy()

    for t in range(num_periods-1):
        u[(t+1)*period_length:(t+2)*period_length, :] = u[:period_length, :].copy()
        y[(t+1)*period_length:(t+2)*period_length, :] = y[:period_length, :].copy()
    return u, y


def define_dim(u, getlist=None):
    n = u.shape[1]
    list_col = []
    for i in range(u.shape[1]):
        s = sum(u[:, i])
        if s == 0 or s == u.shape[0]:
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


def graph_simple_sequence3(N, timestamps):
    u = np.zeros((timestamps - 1, int(N * (N - 1) / 2)))
    y = np.zeros((timestamps - 1, int(N * (N - 1) / 2)))
    u[0, :1] = 1

    s = 1
    k = 0
    kk = 0
    tt = 0
    pp = 0
    chk = True

    for t in range(0, timestamps - 1):
        y[t, :] = u[t, :].copy()
        if sum(u[t, :]) == 0:
            chk = False
            if kk >= 6:
                s = 1
                chk = True
                kk = 0
            else:
                kk += 1
        elif sum(u[t, :]) == len(u[t, :]):
            chk = False
            if k >= 3:
                s = -1
                chk = True
                k = 0
            else:
                k += 1
        elif sum(u[t, :]) == int(len(u[t, :]) / 2):
            chk = False
            if tt >= 2:
                tt = 0
                chk = True
                if pp % 2 == 1:
                    s = -s
                pp += 1

                # if t < timestamps/2:
                #    s = -s
            else:
                tt += 1

        if chk:
            if s == 1:
                for j in range(y.shape[1]):
                    if y[t, j] == 0:
                        y[t, j] = 1
                        break
            if s == -1:
                for j in reversed(range(y.shape[1])):
                    if y[t, j] == 1:
                        y[t, j] = 0
                        break

        if t != timestamps - 2:
            u[t + 1, :] = y[t, :].copy()
    return u, y


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

