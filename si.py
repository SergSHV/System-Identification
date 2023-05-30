from utilities import *
import math
from tqdm.notebook import tqdm

from nfoursid.nfoursid import NFourSID
from nfoursid.utils import Utils

max_err = 1e10


def simulation(inp, state_space, time, threshold=None, x_out=False):
    """
    Create a sequence of outputs (length='time') based on the input data and SI model

    :param inp: input data
    :param state_space: SI model
    :param time: length of time sequence
    :param threshold: rounding scheme (if any)
    :param x_out: vector of internal states
    :return:
    :rtype: output sequence (with/without internal states)
    """

    # SI parameters
    a = state_space.a
    b = state_space.b
    c = state_space.c
    d = state_space.d
    x = state_space.xs

    if len(x) == 0:
        x = np.zeros((a.shape[0], 1))

    # dimension of output and input vectors
    out_dim, in_dim = d.shape[0], d.shape[1]

    # initialize the output
    out = np.zeros((time, out_dim))

    # initialize input states
    states = np.zeros((time+1, a.shape[0])) if x_out else []

    # generate the output sequence
    u = inp.reshape((len(inp), 1))
    if d.shape[0] == len(inp):
        for i in range(time):
            out[i, :] = (np.matmul(c, x) + np.matmul(d, u))[:, 0]
            if threshold is not None:
                for j in range(out_dim):
                    out[i, j] = 1 if out[i, j] >= threshold else 0
            x = np.matmul(a, x) + np.matmul(b, u)
            if x_out:
                states[i+1, :] = x.reshape((1, a.shape[0])).copy()
            u = out[i, :].reshape((in_dim, 1))

    if x_out:
        return out, states
    else:
        return out


def initial_state(x, s, state_space, u, y, mode=None):
    """
    Identify the initial internal state

    :param x: internal state x at time s
    :param s: time
    :param state_space: SI model
    :param u: input data
    :param y: output data
    :param mode: type of identification
    :return: internal state x at time 0
    """

    a = state_space.a
    b = state_space.b

    # Identify the initial internal state
    if mode is None:  # 1st method (based on x[s] and matrices A,B only)
        p_a = np.linalg.pinv(a)
        for i in range(s):
            x = p_a @ (x - b @ u[s - 1 - i, :])
    elif state_space.d.shape[0] == u.shape[1]:
        c = state_space.c
        d = state_space.d

        if mode == 0:  # 2nd method (based on u[s] and y[s] equation)
            p_a = np.linalg.pinv(a)
            v = y[s, :] - d @ u[s, :]
            v2 = 0
            for i in range(1, s + 1):
                v2 += np.linalg.matrix_power(a, i - 1) @ b @ u[s - i, :]
            v2 = c @ v2
            x = np.linalg.matrix_power(p_a, s) @ np.linalg.pinv(c) @ (v - v2)
        else:
            abcd = np.concatenate([np.concatenate([a, b], axis=1), np.concatenate([c, d], axis=1)])
            abcd = np.linalg.matrix_power(abcd, s)
            size_a = a.shape[0]
            if mode == 1:  # the input state equation from [x_s,u_s] = Q^s * [x_0,u_0]
                x = np.linalg.pinv(abcd[size_a:, :size_a]) @ (u[s, :] - abcd[size_a:, size_a:] @ u[0, :])
            elif mode == 2:  # the whole state equation [x_s,u_s] = Q^s * [x_0,u_0]
                x = np.linalg.lstsq(abcd[:, :size_a],
                                    np.concatenate([x, u[s, :]]) - abcd[:, size_a:] @ u[0, :], rcond=None)[0]
            elif 3 <= mode <= 4:  # estimate the best input state based on initial observation u[0], matrices ABCD and y
                vec_b = (np.concatenate([x, u[s, :]]) - abcd[:, size_a:] @ u[0, :])[size_a:]
                vec_a = abcd[size_a:, :size_a]
                k_abcd = np.concatenate([np.concatenate([a, b], axis=1), np.concatenate([c, d], axis=1)])

                max_v = s if mode == 4 else y.shape[0]
                for k in range(1, max_v):
                    pow_m = np.linalg.matrix_power(k_abcd, k).copy()
                    vec_b = np.concatenate(
                        [vec_b, y[k, :] - pow_m[size_a:, size_a:] @ u[0, :]], axis=0)
                    vec_a = np.concatenate([vec_a, pow_m[size_a:, :size_a]], axis=0)
                x = np.linalg.lstsq(vec_a, vec_b, rcond=None)[0]
            elif mode == 5:  # https://elib.dlr.de/28249/1/sima-varga-rasp-ident.pdf  Page 61
                y0 = simulation(u[0, :].reshape((u.shape[1], 1)), state_space, u.shape[0])
                vec_a = c.copy()  # matrix C
                for k in range(1, y.shape[0]):
                    vec_a = np.concatenate([vec_a, c @ np.linalg.matrix_power(a, k).copy()])
                x = np.linalg.lstsq(vec_a, (y-y0).flatten(), rcond=None)[0]
    else:
        x = np.zeros(a.shape[0])

    return x


def simulation_error(state_space, u, y, thresholds):
    """
    Compute the SI model error based on the graph data simulation
    :param state_space: SI models
    :param u: input data
    :param y: output data
    :param thresholds: threshold values to consider
    :return: SI model error and best rounding scheme
    """
    loss = [float("inf"), float("inf"), float("inf")]
    best_thr = np.zeros(3)
    inp = u[0, :].reshape((u.shape[1], 1))

    # Compute the MSE loss for a given SI model (no rounding)
    out_raw = simulation(inp, state_space, y.shape[0])
    loss[0] = mse(out_raw, y)

    # Compute the MSE loss for a given SI model (rounding at the end)
    for thr in range(len(thresholds)):
        err = mse(rounding(out_raw, thresholds[thr] / 100), y)
        if err < loss[2] and not math.isclose(err, loss[2]):
            loss[2] = err
            best_thr[2] = thresholds[thr] / 100

    # Compute the MSE loss for a given SI model (rounding at each step)
    for thr in range(len(thresholds)):
        out_rounded = simulation(inp, state_space, y.shape[0], thresholds[thr] / 100)
        err = mse(out_rounded, y)
        if err < loss[1] and not math.isclose(err, loss[1]):
            loss[1] = err
            best_thr[1] = thresholds[thr] / 100

    return loss, best_thr


def io_error(state_space, u, y, thresholds):
    """
    Compute the SI model error based on the input-output data
    :param state_space: SI models
    :param u: input data
    :param y: output data
    :param thresholds: threshold values to consider
    :return: SI model error and best rounding scheme
    """
    loss = [0, float("inf")]
    best_thr = np.zeros(2)
    loss_thr = np.zeros(len(thresholds))

    for i in range(y.shape[0]):
        inp = u[i, :].reshape((u.shape[1], 1))
        out = simulation(inp, state_space, 1, threshold=None)

        # No rounding
        loss[0] += mse(out, y[i, :]) ** 2

        # Compute the MSE loss for a given SI model (rounding at each step)
        for thr in range(len(thresholds)):
            out_rounded = rounding(out, thresholds[thr] / 100)
            loss_thr[thr] += mse(out_rounded, y[i, :]) ** 2

        if loss[0] > max_err:
            break

    loss[0] = loss[0] ** (1 / 2)

    # find the best threshold
    for thr in range(len(thresholds)):
        if loss_thr[thr] <= loss[1] and not math.isclose(loss_thr[thr], loss[1]):
            loss[1] = loss_thr[thr]
            best_thr[1] = thresholds[thr] / 100

    loss[1] = loss[1] ** (1 / 2)

    return loss, best_thr


def compute_error(param_list, u, y, thresholds=None, mode="sim", silent=True):
    """
    Compute error for all SI models
    :param param_list: set of SI models
    :param u: input vector
    :param y: output vector
    :param thresholds: threshold values to consider
    :param mode: how to compute the error: based on the input/output data or on the simulated sequence
    :param silent: progress bar visibility
    :return:
    """
    err_list = np.zeros((len(param_list), 3))
    thr_list = np.zeros((len(param_list), 3))

    if thresholds is None:
        thresholds = list(range(40, 51, 3))

    for p in tqdm(range(len(param_list)), disable=silent):
        if mode == "sim":
            err_list[p, :], thr_list[p, :] = simulation_error(param_list[p][2], u, y, thresholds)
        elif mode == "io":
            err, thr = io_error(param_list[p][2], u, y, thresholds)
            err_list[p, :], thr_list[p, :] = err.copy(), thr.copy()

    return err_list, thr_list


def best_error(err_list, thr_list):
    """
    Find the best parameters for the model
    :param err_list: error list for SI models
    :param thr_list: best thresholds for SI models
    :return: indices for best models
    """
    best_ind = [0]*3
    best_err, best_thr = err_list[0, :].copy(), thr_list[0, :].copy()

    # write the SI model parameters with the lowest loss
    for j in range(err_list.shape[1]):
        for i in range(1, err_list.shape[0]):
            if err_list[i, j] < best_err[j] and not math.isclose(err_list[i, j], best_err[j]):
                best_err[j] = err_list[i, j].copy()
                best_ind[j] = i

    # write the best threshold
    for j in range(err_list.shape[1]):
        best_thr[j] = thr_list[best_ind[j], j].copy()

    return best_ind, best_thr, best_err


def visualise(u, y, state_space, mode, thresholds=None, figure_size=None):
    """
    Visualize simulated graph sequences
    :param u: input vector
    :param y: output vector
    :param state_space: SI model
    :param mode: type of the model (rounding/no rounding)
    :param thresholds: the best thresholds for SI models
    :param figure_size:plot size of figures
    :return: graph sequences and internal states
    """
    out, x_states = [], []

    if figure_size is None:
        figure_size = (12, 3)

    if mode == 0:  # No rounding
        out, x_states = simulation(u[0, :].reshape((u.shape[1], 1)), state_space[2], u.shape[0], x_out=True)
    elif mode == 1:  # Rounding at each step
        out, x_states = simulation(u[0, :].reshape((u.shape[1], 1)), state_space[2], u.shape[0],
                                   threshold=thresholds[mode], x_out=True)
    elif mode == 2:  # Rounding at the end
        out, x_states = simulation(u[0, :].reshape((u.shape[1], 1)), state_space[2], u.shape[0], x_out=True)
        out = rounding(out, threshold=thresholds[mode])

    print("The total loss is ", mse(out, y))
    # pd.DataFrame(cum_mse(out_raw, y)).plot()
    pd.DataFrame(y).plot.line(title='Values of adjacency vector (Real)', figsize=figure_size)
    pd.DataFrame(out).plot.line(title='Values of adjacency vector (Model)', figsize=figure_size)
    plt.axvline(x=state_space[0], ymin=0, ymax=1, color='purple', ls='--')
    pd.DataFrame(y - out).plot.line(title='Real output - Model: Difference', figsize=figure_size)
    plt.axvline(x=state_space[0], ymin=0, ymax=1, color='purple', ls='--')
    pd.DataFrame(x_states[:, :]).plot.line(title='Values of state X', figsize=figure_size)

    return out, x_states


def compare_loss(outs, y, labels=None, figure_size=None):
    mse_err = np.zeros((y.shape[0], len(outs)))

    if figure_size is None:
        figure_size = (12, 3)

    if labels is None:
        labels = ['No rounding', 'Rounding at each step', 'Rounding at the final stage']

    for i in range(y.shape[0]):
        for j in range(len(outs)):
            mse_err[i, j] = mse(outs[j][:i + 1, :], y[:i + 1, :])

    pd.DataFrame(mse_err, columns=labels).plot.line(title='Cumulative MSE', figsize=figure_size)
    return mse_err[-1, :]


def compute_all_abcd(u, y, x_mode=None, silent=True):
    param_list = []
    s = int((u.shape[0]) / (u.shape[1] + y.shape[1] + 2))+1
    t = define_periodicity(u)

    for ns in tqdm(range(1, min(s, t)), disable=silent):
        nfoursid = NFourSID(pd.DataFrame(np.concatenate((u, y), axis=1), columns=int_list(u.shape[1] + y.shape[1])),
                            input_columns=int_list(u.shape[1]), num_block_rows=ns,
                            output_columns=int_list(u.shape[1] + y.shape[1])[u.shape[1]:])
        u_hankel = Utils.block_hankel_matrix(nfoursid.u_array, nfoursid.num_block_rows)
        y_hankel = Utils.block_hankel_matrix(nfoursid.y_array, nfoursid.num_block_rows)

        u_past, u_future = u_hankel[:, :-nfoursid.num_block_rows], u_hankel[:, nfoursid.num_block_rows:]
        y_past, y_future = y_hankel[:, :-nfoursid.num_block_rows], y_hankel[:, nfoursid.num_block_rows:]
        u_instrumental_y = np.concatenate([u_future, u_past, y_past, y_future])

        q, r = map(lambda matrix: matrix.T, np.linalg.qr(u_instrumental_y.T, mode='reduced'))

        y_rows, u_rows = nfoursid.y_dim * nfoursid.num_block_rows, nfoursid.u_dim * nfoursid.num_block_rows
        nfoursid.R32 = r[-y_rows:, u_rows:-y_rows]
        nfoursid.R22 = r[u_rows:-y_rows, u_rows:-y_rows]
        nfoursid.R32_decomposition = Utils.eigenvalue_decomposition(nfoursid.R32)

        # [M.Verhaegen & V.Verdult 2007], page 333
        order = np.linalg.matrix_rank(nfoursid.R32 @ np.linalg.pinv(nfoursid.R22) @ np.concatenate([u_past, y_past]))
        state_space_identified = identify_system(nfoursid, order, s, u, y, x_mode)
        if state_space_identified is not None:
            param_list.append((ns, order, state_space_identified))
    return param_list


def compute_all_abcd_old(u, y, x_mode=None, silent=True, n_order=None):
    param_list = []
    s = (u.shape[0]) / (u.shape[1] + y.shape[1] + 2)
    t = define_periodicity(u)

    for s in tqdm(range(1, min(int(s) + 1, t)), disable=silent):
        nfoursid = NFourSID(pd.DataFrame(np.concatenate((u, y), axis=1), columns=int_list(u.shape[1] + y.shape[1])),
                            input_columns=int_list(u.shape[1]), num_block_rows=s,
                            output_columns=int_list(u.shape[1] + y.shape[1])[u.shape[1]:])
        nfoursid.subspace_identification()
        ev = np.diagonal(nfoursid.R32_decomposition.eigenvalues)

        if n_order is not None:
            if n_order <= ev[ev > 0].shape[0]+1:
                state_space_identified = identify_system(nfoursid, n_order, s, u, y, x_mode)
                if state_space_identified is not None:
                    param_list.append((s, n_order, state_space_identified))
        else:
            for n in range(1, ev[ev > 0].shape[0]+1):
                state_space_identified = identify_system(nfoursid, n, s, u, y, x_mode)
                if state_space_identified is not None:
                    param_list.append((s, n, state_space_identified))
    return param_list


def identify_system(nfoursid, order, s, u, y, x_mode):
    state_space_identified = nfoursid.system_identification(rank=order)[0]
    obs_dec = nfoursid._get_observability_matrix_decomposition()
    x = initial_state((np.power(obs_dec.eigenvalues, .5) @ obs_dec.right_orthogonal)[:, 0],
                      s, state_space_identified, u, y, x_mode)
    state_space_identified.xs = x.reshape((len(x), 1))
    return state_space_identified if state_space_identified.d.shape[0]==u.shape[1] else None

