
from utilities import *


def generate_g(n, num_periods=20, per_length=None, changes=1):
    period_length = n*(n-1) if per_length is None else per_length
    dim = int(n * (n - 1) / 2)

    order_inc, order_dec = generate_edge_order(dim, order="fixed")
    u, y = create_initial_u_y(dim, timestamps=num_periods*period_length, num_ones=int(dim/2), order=order_inc)

    for t in range(period_length):
        y[t, :] = u[t, :].copy()

        for ch in range(changes):
            curr_dist = np.count_nonzero(y[t, :] != u[0, :])
            remained_changes = (period_length-t-1)*changes + (changes - 1 - ch) + 1

            if curr_dist < remained_changes - 1:
                p_add = 1 if sum(y[t, :]) != len(y[t, :]) else 0
                p_remove = 1 if sum(y[t, :]) != 0 else 0
                p_keep = (p_add+p_remove)/4 if (p_add+p_remove) != 0 else 1
                p_sum = p_add + p_remove + p_keep
                action = np.random.choice(3, p=[p_remove/p_sum, p_keep/p_sum, p_add/p_sum])  # 0=remove,1=keep,2=add
                if action == 0:  # remove edge
                    l_edges = []
                    for k in range(u.shape[1]):
                        if y[t, k] == 1:
                            l_edges.append(k)
                    edge = np.random.choice(l_edges)
                    y[t, edge] = 0

                if action == 2:  # add edge
                    l_edges = []
                    for k in range(u.shape[1]):
                        if y[t, k] == 0:
                            l_edges.append(k)
                    edge = np.random.choice(l_edges)
                    y[t, edge] = 1
            else:
                l_add = []
                l_remove = []
                l_keep = []
                for k in range(u.shape[1]):
                    a1 = np.concatenate([u[0, :k], u[0, k+1:]])
                    a2 = np.concatenate([y[t, :k], y[t, k+1:]])
                    new_d = np.count_nonzero(a1 != a2)
                    if curr_dist == new_d and new_d < remained_changes:
                        l_keep.append(k)
                    if curr_dist > new_d:
                        if y[t, k] == 0:
                            l_add.append(k)
                        if y[t, k] == 1:
                            l_remove.append(k)
                p_add = 1 if sum(y[t, :]) != len(y[t, :]) and len(l_add) > 0 else 0
                p_remove = 1 if sum(y[t, :]) != 0 and len(l_remove) > 0 else 0

                p_keep = 0
                if len(l_keep) > 0:
                    p_keep = (p_add+p_remove)/4
                    if p_keep == 0:
                        p_keep = 1
                p_sum = p_add + p_remove + p_keep
                action = np.random.choice(3, p=[p_remove/p_sum, p_keep/p_sum, p_add/p_sum])  # 0=remove,1=keep,2=add
                if action == 0:  # remove edge
                    edge = np.random.choice(l_remove)
                    y[t, edge] = 0

                if action == 2:  # add edge
                    edge = np.random.choice(l_add)
                    y[t, edge] = 1

        if t != period_length-1:
            u[t + 1, :] = y[t, :].copy()

    for t in range(num_periods-1):
        u[(t+1)*period_length:(t+2)*period_length, :] = u[:period_length, :].copy()
        y[(t+1)*period_length:(t+2)*period_length, :] = y[:period_length, :].copy()
    return u, y


def graph_simple_sequence(n, timestamps, num_ones=None, s=None, order="fixed", dim=None):
    """
    :param n: number of nodes
    :param timestamps: period length
    :param num_ones: number of 1 in the initial graph G0
    :param s: type of action (0 - remove edge, 1 - add edge)
    :param order: order of edges (fixed, random)
    :param dim: order of edges having periodic dynamic
    :return:
    """
    dim = int(n * (n - 1) / 2) if dim is None else dim
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


def graph_simple_sequence2(n, timestamps, num_ones=None, s=None, constant=None, order="fixed"):
    """
    :param n: number of nodes
    :param timestamps: period length
    :param num_ones: number of 1 in the initial graph G0
    :param s: type of action (0 - remove edge, 1 - add edge)
    :param constant: number of periods to keep the graph
    :param order: how add edges (fixed edge sequence or random edge sequence)
    :return:
    """

    if constant is None:
        constant = [6, 3, 2]

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


def graph_simple_sequence3(n, timestamps):
    u = np.zeros((timestamps - 1, int(n * (n - 1) / 2)))
    y = np.zeros((timestamps - 1, int(n * (n - 1) / 2)))
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


def generate_edge_order(dim, order):
    order_inc = list(range(dim))
    order_dec = list(range(dim))
    if order == "random":
        order_inc = np.random.permutation(dim)
        order_dec = np.random.permutation(dim)
    return order_inc, order_dec


def generate_noise_data(size, num_rand, periods):
    timestamps = int((size * (size - 1))) * periods
    u, y = graph_simple_sequence(size, timestamps, 1, 1)

    u = np.concatenate([u, np.zeros((u.shape[0], num_rand))], axis=1)
    y = np.concatenate([y, np.zeros((y.shape[0], num_rand))], axis=1)

    for t in range(0, u.shape[0]):
        for i in range(num_rand):
            y[t, -(i + 1)] = np.random.choice([0, 1], size=(1, 1), p=[1 / 2, 1 / 2])
        if t != u.shape[0] - 1:
            u[t + 1, -num_rand:] = y[t, -num_rand:].copy()
    return u, y


def generate_noise_data2(size, periods, num_changes=2):
    timestamps = int((size * (size - 1))) * periods
    u, y = graph_simple_sequence(size, timestamps, 1, 1)
    for i in range(u.shape[0]):
        ind_list = np.random.randint(u.shape[1], size=num_changes)
        for ind in ind_list:
            y[i, ind] = 1 - y[i, ind]
        if i + 1 != u.shape[0]:
            u[i + 1, :] = y[i, :].copy()
        else:
            u[0, :] = y[i, :].copy()
    return u, y
