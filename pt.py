from utilities import define_periodicity, mse, average_period, rounding, norm_n, tqdm
from matrix_identification import lpg_gen
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx


def find_local_min(loss):
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


def loc_min_alg(loss, max_per=None, ex_list=None):
    """
    Determine the best period
    :param loss: mse sequence
    :param max_per: limit of the maximal period
    :param ex_list: exception list of periods
    :return: best period
    """
    n = len(loss)
    l_min = find_local_min(loss)
    best_order = 0
    best_ratio = 0
    best_loss = float('inf')
    if max_per is None:
        max_per = n / 2
    if ex_list is None:
        ex_list = []

    for loc in l_min:
        if loc < max_per and loc != 0 and loc not in ex_list:
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
            # elif ratio == best_ratio and avg_loss < best_loss:
            #    best_ratio = ratio
            #    best_loss = avg_loss
            #    best_order = loc
    return best_order, best_ratio


def estimate_periodicity(u, max_period, per_range, figures, ex_list=None, silent=False):
    """
    Define the best period for the periodicity transform (paper "System Identification for Temporal Networks")
    :param u: initial sequence
    :param max_period: max period for periodicity transform
    :param per_range: range of mse to define the best period
    :param figures: draw loss figure
    :param ex_list: exception list for period length
    :param silent: progress bar visibility
    :return: optimal period for PT, captured loss
    """
    min_p, max_p = per_range[0], per_range[1]

    ttt, uuu, loss = define_periodicity(u, min_period=min_p, max_period=max_p, mode="noise", silent=silent)
    best_period, best_ratio = loc_min_alg(loss, max_period, ex_list)

    if figures:
        plt.plot(range(1, len(loss) + 1), loss)
        plt.xlabel("Period Length")
        plt.ylabel("Loss (MSE)")
        plt.show()
        print(f"Best period is {best_period}, it is a local minima in {best_ratio} cases")

    return best_period, loss[0] - loss[best_period]


def lg_gen(u, l_num, max_period=None, per_range=None, modelling=None, figures=None, silent=False):
    """
    :param u:
    :param l_num:
    :param modelling:
    :param max_period: max period for periodicity transform
    :param per_range:
    :param silent: progress bar visibility
    :param figures: it
    """
    if max_period is None:
        max_period = u.shape[0] / 2
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

    for i in range(l_num):
        print("Stage %s started..." % (i + 1))
        best_period, loss_capt = estimate_periodicity(u_copy, max_period=max_period,
                                                      per_range=per_range, figures=figures, silent=silent)
        rem_loss -= loss_capt
        avg_patterns.append(average_period(u_copy, best_period))
        if modelling:
            qx.append(lpg_gen(avg_patterns[-1], []))

        loss_arrays.append(loss_capt)
        u_copy -= np.tile(avg_patterns[-1], (u_copy.shape[0] // avg_patterns[-1].shape[0] + 1, 1))[:u_copy.shape[0], :]
        print("Stage %s finished. MSE captured is %s. Remained loss is %s" % ((i + 1), loss_arrays[-1], rem_loss))

    return avg_patterns, loss_arrays, qx


def generate_crit_arr(u_copy, criteria):
    """
    Aggregate Data for the Period Identification
    :param u_copy: initial array
    :param criteria: method of aggregation
    :return: output sequence
    """
    u_crit = []
    if criteria == "full data":
        u_crit = u_copy
    if criteria == "link count":
        u_crit = np.sum(u_copy, axis=1).reshape(u_copy.shape[0], 1)
    if criteria == "link difference":
        u_crit = np.zeros((u_copy.shape[0] - 1, 1))
        for k in range(u_copy.shape[0] - 1):
            u_crit[k, 0] += sum(abs(u_copy[k, :] - u_copy[k + 1, :]))
    return u_crit


def small2large(arr, threshold, start, max_period, rem_loss):
    chk = True
    best_period = 0
    while chk and start <= max_period:
        pattern = average_period(arr, start)
        new_mse = mse(arr - np.tile(pattern, (arr.shape[0] // pattern.shape[0] + 1, 1))[:arr.shape[0], :],
                      np.zeros((1, arr.shape[1])))
        if new_mse < threshold * rem_loss:
            best_period = start
            chk = False
        start += 1
    return best_period


def period_rec_dft(x, period_list, max_period):
    per = 0
    ind_list = period_dft(x, max_period)

    for i in range(len(ind_list)):
        if ind_list[-(i + 1)] != len(x) and ind_list[-(i + 1)] not in period_list:
            per = ind_list[-(i + 1)]
            period_list.append(per)
            break
    return per, period_list


def period_dft(x, max_period):
    n = (x.shape[0] // 2) + 1 if x.shape[0] % 2 == 0 else (x.shape[0] + 1) // 2
    freq = np.zeros((n, x.shape[1]))
    for i in range(x.shape[1]):
        freq[:, i] = abs(np.fft.rfft(x[:, i]))

    arg_list = np.argsort(np.sum(freq, axis=1))
    ind_list = []
    for i in range(len(arg_list)):
        if arg_list[i] == 0:
            ind_list.append(1)
        elif arg_list[i] > 1:
            v = int(np.round(len(x) / arg_list[i]))
            if v <= max_period:
                ind_list.append(v)
    indexes = np.unique(ind_list, return_index=True)[1]
    ind_list = [ind_list[index] for index in sorted(indexes)]

    return ind_list


def best_correlation(arr, max_period, exc_list, silent):
    max_cor = 0
    per = 0
    for p in tqdm(range(1, max_period+1), disable=silent):
        if p not in exc_list:
            for s in range(0, p):
                corr = sum(abs(np.sum(arr.T[:, s::p], axis=1)))
                if corr > max_cor:
                    max_cor = corr
                    per = p
    return per


def get_factors(n):
    factors = []
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def m_best(arr, m, max_period, gamma=False):
    arr_copy = arr.copy()
    periods = np.zeros(m, dtype=np.uint32)
    norms = np.zeros(m)
    pattern_list = []

    # step 1
    for i in range(m):
        best_norm = 0
        best_period = 0
        best_pattern = None
        for p in range(1, max_period + 1):
            pattern = average_period(arr_copy, p)
            norm = norm_n(pattern, arr.shape[0])

            if gamma:
                norm /= p ** (1 / 2)
            if norm > best_norm:
                best_norm = norm
                best_period = p
                best_pattern = pattern.copy()
        pattern_list.append(best_pattern)
        norms[i] = best_norm
        periods[i] = best_period

        arr_copy -= np.tile(best_pattern, (arr.shape[0] // best_pattern.shape[0] + 1, 1))[:arr.shape[0], :]

    # step 2
    changed = True
    while changed:
        i = 0
        while i < m:
            changed = False
            best_norm = 0
            best_period = 0
            best_pattern = []
            factors = get_factors(periods[i])

            for f in factors:
                pattern = average_period(pattern_list[i], f)
                norm = norm_n(pattern, arr.shape[0])
                # pattern = np.tile(pattern, (arr.shape[0] // pattern.shape[0] + 1, 1))[:arr.shape[0], :]
                # norm = mse(pattern, np.zeros((1, arr.shape[1])))
                if gamma:
                    norm /= f ** (1 / 2)
                if norm > best_norm:
                    best_norm = norm
                    best_period = f
                    best_pattern = pattern.copy()

            if best_period not in periods and best_period != 0:
                x_big_q = np.tile(best_pattern, (pattern_list[i].shape[0] // best_pattern.shape[0] + 1,
                                                 1))[:pattern_list[i].shape[0], :]
                xq = pattern_list[i] - x_big_q
                n_big_q = best_norm
                nq = norm_n(xq, arr.shape[0])
                if gamma:
                    nq /= best_period ** (1 / 2)

                min_q = min(norms)
                if (nq + n_big_q) > (norms[m - 1] + norms[i]) and (nq > min_q) and (n_big_q > min_q):
                    changed = True
                    # keep the old one but now it's weakened. Replace values.

                    pattern_list = pattern_list[:i] + [best_pattern] + [xq] + pattern_list[i+1:]

                    # pattern_list[i] = xq.copy()
                    # pattern_list = np.insert(pattern_list, i, best_pattern, 0)

                    norms[i] = nq
                    norms = np.insert(norms, i, n_big_q)

                    periods = np.insert(periods, i, best_period)

                    pattern_list = pattern_list[:m]
                    norms = norms[:m]
                    periods = periods[:m]
                else:
                    i += 1
            else:
                i += 1

    return periods


# The relevant paper is:
# [] S.V. Tenneti and P. P. Vaidyanathan, "Nested Periodic Matrices and Dictionaries:
# New Signal Representations for Period Estimation", IEEE Transactions on Signal
# Processing, vol.63, no.14, pp.3736-50, July, 2015.
def create_dictionary(p_max, row_size, method):
    arr = []
    for N in range(1, p_max + 1):
        cn_arr = []
        if method == 'Ramanujan':
            k = []  # coprime number to N
            for kk in range(1, N + 1):
                if np.gcd(kk, N) == 1:
                    k.append(kk)

            c1 = np.zeros(N)
            for n in range(N):
                for a in k:
                    c1[n] += np.exp(1j * 2 * np.pi * a * n / N)
            c1 = np.real(c1)
            cn_col_size = len(k)
            cn_arr = c1.reshape((N, 1))
            for i in range(1, cn_col_size):
                cn_arr = np.concatenate([cn_arr, np.roll(c1, i).reshape((N, 1))], axis=1)
        elif method == 'Farey':
            a_dft = np.fft.fft(np.eye(N))
            a = np.array(range(N))
            a[0] = N
            a = N / np.gcd(a, N)
            i_arr = np.array(range(N)).astype(int)
            i_arr = i_arr[a[i_arr] == N]
            cn_arr = a_dft[:, i_arr]

        cna_arr = np.tile(cn_arr, (row_size // N, 1))
        cn_cutoff_arr = cn_arr[:(row_size % N), :]
        cna_arr = np.concatenate([cna_arr, cn_cutoff_arr], axis=0)

        if N == 1:
            arr = cna_arr
        else:
            arr = np.concatenate([arr, cna_arr], axis=1)
    arr = np.round(arr)

    for i in range(arr.shape[1]):
        arr[:, i] /= np.linalg.norm(arr[:, i])

    return arr


def strength_vs_period_l1(x, a_arr, p_max):
    # Strength_vs_Period_L1(x,p_max,method) plots the strength vs period plot
    # for the signal x using an L1 norm based convex program.

    # Penalty Vector Calculation
    penalty_vector = create_penalty(p_max)

    s = cvx.Variable(a_arr.shape[1])  # b is dim x

    objective = cvx.Minimize(cvx.norm(s @ penalty_vector, 1))  # L_1 norm objective function
    constraints = [a_arr @ s == x]  # y is dim a and M is dim a by b

    cvx.Problem(objective, constraints).solve(verbose=False, solver='SCS')

    # then clean up and chop the 1e-12 vals out of the solution
    s = np.array(s.value)

    return create_energy(s, a_arr, len(x), p_max)


def create_penalty(p_max):
    penalty_vector = []
    for i in range(1, p_max + 1):
        k_red = 0  # coprime number to N
        for kk in range(1, i + 1):
            if np.gcd(kk, i) == 1:
                k_red += 1
        if i == 1:
            penalty_vector = i * np.ones((k_red, 1))
        else:
            penalty_vector = np.concatenate([penalty_vector, i * np.ones((k_red, 1))], axis=0)

    penalty_vector = np.power(penalty_vector, 2)
    return penalty_vector


def create_energy(s, a_arr, len_x, p_max):
    energy_s = np.zeros(p_max)
    patterns_s = np.zeros((len_x, p_max))
    current_index_end = 0
    for i in range(1, p_max + 1):
        k_red = 0  # coprime number to N
        for kk in range(1, i + 1):
            if np.gcd(kk, i) == 1:
                k_red += 1
        current_index_start = current_index_end
        current_index_end = current_index_end + k_red
        for j in range(current_index_start, current_index_end):
            energy_s[i - 1] += ((abs(s[j])) ** 2)
            patterns_s[:, i - 1] += np.real(s[j] * a_arr[:, j])
    energy_s[0] = 0

    return energy_s, patterns_s


def strength_vs_period_l2(x, a_arr, p_max):
    # Strength_vs_Period_L1(x,p_max,method) plots the strength vs period plot for the signal x
    # using an L2 norm based convex program.

    # Penalty Vector Calculation
    penalty_vector = create_penalty(p_max)
    d_arr = np.diag(np.power(1. / penalty_vector, 2).flatten())

    pp_arr = d_arr @ np.transpose(a_arr) @ np.linalg.inv(a_arr @ d_arr @ np.transpose(a_arr))
    s = pp_arr @ x

    return create_energy(s, a_arr, len(x), p_max)


def ramanujan_pt(arr, max_period, criteria):
    energy = np.zeros(max_period)
    freq = np.zeros((arr.shape[1], arr.shape[0], max_period))

    if "Farey" in criteria:
        a_dict = create_dictionary(max_period, arr.shape[0], 'Farey')
    else:
        a_dict = create_dictionary(max_period, arr.shape[0], 'Ramanujan')
    #print("Array is ready")
    for i in range(arr.shape[1]):
        if "L1" in criteria:
            out = strength_vs_period_l1(arr[:, i], a_dict, max_period)
        else:
            out = strength_vs_period_l2(arr[:, i], a_dict, max_period)
        energy = np.add(energy, out[0])
        freq[i, :, :] = out[1]
    return (np.argsort(energy) + 1), freq


def periodicity_transformation(u, l_num, mode, criteria=None, max_period=None,
                               per_range=None, params=None, figures=None, silent=True, per_list=None):
    dic, dic_per = {}, {}
    u_copy = u.copy()
    avg_patterns = []
    if figures is None:
        figures = False
    param = None

    initial_mse = mse(u, np.zeros((1, u.shape[1])))
    crit_mse = 0

    list_mse = [initial_mse]
    list_mse_rounding = [initial_mse]
    list_freq = []

    period_list = []
    u_crit = generate_crit_arr(u_copy, criteria)
    if mode == "PT (DFT)":
        period_list = period_dft(u_crit, max_period)
    elif mode == "l-best":
        period_list = m_best(u_crit, l_num, max_period)
    elif mode == "l-best-gamma":
        period_list = m_best(u_crit, l_num, max_period, True)
    elif "PT (Ramanujan)" in mode or "PT (Farey)" in mode:
        period_list, freq = ramanujan_pt(u_crit, max_period, mode)

    start = 1

    for i in range(l_num):
        best_period = 0
        if i != 0:
            u_crit = generate_crit_arr(u_copy, criteria)

        if mode == "LocalMin":
            if per_list is None:
                best_period, _ = estimate_periodicity(u_crit, max_period=max_period, ex_list=list_freq,
                                                  per_range=per_range, figures=figures, silent=silent)
            else:
                best_period = per_list[i]
        elif mode == "Small2Large":
            param = params[1] + ', ' + str(params[0])
            if i == 0 or params[1] == "residual":
                crit_mse = mse(u_crit, np.zeros((1, u_crit.shape[1])))
            best_period = small2large(u_crit, params[0], start, max_period, crit_mse)
            if best_period != 0:
                start = best_period + 1
        elif mode == "Best correlation":
            best_period = best_correlation(u_crit, max_period, list_freq, silent)
        elif mode == "l-best" or mode == "l-best-gamma":
            best_period = period_list[i]

        elif mode == "PT (DFT)":
            if i < len(period_list):
                best_period = period_list[-(i + 1)]
        elif mode == "PT (iterative DFT)":
            best_period, period_list = period_rec_dft(u_crit, period_list, max_period)
        elif "iterative" in mode:
            period_list, freq = ramanujan_pt(u_crit, max_period, mode)
            best_period = period_list[-1]
            j = 0
            while best_period in list_freq:
                j += 1
                best_period = period_list[-(j+1)]

        elif "PT (Ramanujan)" in mode or "PT (Farey)" in mode:
            best_period = period_list[-(i + 1)]

        if len(list_freq) > 0 and best_period == list_freq[-1]:
            best_period = 0

        if best_period != 0:
            avg_patterns.append(average_period(u_copy, best_period))
            u_copy -= np.tile(avg_patterns[-1],
                              (u_copy.shape[0] // avg_patterns[-1].shape[0] + 1, 1))[:u_copy.shape[0], :]
            rem_loss = mse(u_copy, np.zeros((1, u.shape[1])))
            list_mse.append(rem_loss)
            list_mse_rounding.append(mse(rounding(u - u_copy), u))
            list_freq.append(best_period)
    if params is not None:
        criteria += ", " + str(param)

    dic[mode + f"({criteria})"] = list_mse
    dic[mode + f"({criteria}, rounding)"] = list_mse_rounding
    dic_per[mode + f"({criteria})"] = list_freq
    return dic, dic_per
