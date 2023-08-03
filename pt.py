from utilities import define_periodicity, mse, average_period
from matrix_identification import lpg_gen
import numpy as np
import matplotlib.pyplot as plt


def local_min(loss):
    """
    Identify local minima in the sequence
    :param loss: initial sequence
    :return: id of local minima
    """
    loc_min = []
    for i in range(len(loss)):
        if i == 0:
            if loss[i] <= loss[i + 1]:
                loc_min.append(i)
        elif i == len(loss) - 1:
            if loss[i] <= loss[i - 1]:
                loc_min.append(i)
        else:
            if loss[i] <= loss[i + 1] and loss[i] <= loss[i - 1] and loss[i] < loss[0]:
                loc_min.append(i)
    return loc_min


def alg_period(loss, max_per=None):
    """
    Determine the best period
    :param loss: mse sequence
    :param max_per: limit of the maximal period
    :return: best period
    """
    n = len(loss)
    l_min = local_min(loss)
    best_order = 0
    best_ratio = 0
    best_loss = float('inf')
    if max_per is None:
        max_per = n / 2

    for loc in l_min:
        if loc < max_per and loc != 0:
            c = 1
            avg_loss = 0
            it = loc
            avg_loss += loss[it]
            while it <= n:
                it += loc
                if it in l_min:
                    avg_loss += loss[it]
                    c += 1
            ratio = c / (n // loc)
            avg_loss = avg_loss / (n // loc)
            if ratio > best_ratio:
                best_ratio = ratio
                best_order = loc
                best_loss = avg_loss
            elif ratio == best_ratio and avg_loss < best_loss:
                best_ratio = ratio
                best_loss = avg_loss
                best_order = loc
    return best_order, best_ratio


def estimate_periodicity(u, max_period, per_range, figures):
    """
    Define the best period for the periodicity transform (paper "System Identification for Temporal Networks")
    :param u: initial sequence
    :param max_period: max period for periodicity transform
    :param per_range: range of mse to define the best period
    :param figures: draw loss figure
    :return: optimal period for PT, captured loss
    """
    min_p, max_p = per_range[0], per_range[1]

    ttt, uuu, loss = define_periodicity(u, min_period=min_p, max_period=max_p, mode="noise")
    best_period, best_ratio = alg_period(loss, max_period)

    if figures:
        plt.plot(range(1, len(loss) + 1), loss)
        plt.xlabel("Period Length")
        plt.ylabel("Loss (MSE)")
        plt.show()
        print(f"Best period is {best_period}, it is a local minima in {best_ratio} cases")

    return best_period, loss[0] - loss[best_period]


def lg_gen(u, l, max_period=None, per_range=None, modelling=None, figures=None):
    """
    :param u:
    :param l:
    :param modelling:
    :param max_period: max period for periodicity transform
    :param per_range:
    :param figures: it
    """
    if max_period is None:
        max_period = u.shape[0]/2
    if per_range is None:
        per_range = [1, u.shape[0]]
    if figures is None:
        figures = True
    if modelling is None:
        modelling = False

    u_copy = u.copy()
    avg_patterns, loss_arrays = [], []
    qx = []

    rem_loss = mse(u, np.zeros((1, u.shape[1])))

    print("Initial MSE loss is ", rem_loss)

    for i in range(l):
        print("Stage %s started..." % (i + 1))
        best_period, loss_capt = estimate_periodicity(u_copy, max_period=max_period,
                                                      per_range=per_range, figures=figures)
        rem_loss -= loss_capt
        avg_patterns.append(average_period(u_copy, best_period))
        if modelling:
            qx.append(lpg_gen(avg_patterns[-1], []))

        loss_arrays.append(loss_capt)
        u_copy -= np.tile(avg_patterns[-1], (u_copy.shape[0] // avg_patterns[-1].shape[0] + 1, 1))[:u_copy.shape[0], :]
        print("Stage %s finished. MSE captured is %s. Remained loss is %s" % ((i + 1), loss_arrays[-1], rem_loss))

    return avg_patterns, loss_arrays, qx
