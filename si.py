def simulation(inp, state_space_identified, time, threshold=None, x_out=False):
    A = state_space_identified.a
    B = state_space_identified.b
    C = state_space_identified.c
    D = state_space_identified.d
    x = state_space_identified.xs

    out_dim, in_dim = D.shape[0], D.shape[1]

    out = np.zeros((time, out_dim))
    states = np.zeros((time, A.shape[0]))
    u0 = inp.reshape((len(inp), 1))
    for i in range(time):
        out[i, :] = (np.matmul(C, x) + np.matmul(D, u0))[:, 0]
        if threshold is not None:
            for j in range(out_dim):
                out[i, j] = 1 if out[i, j] >= threshold else 0
        x = np.matmul(A, x) + np.matmul(B, u0)
        states[i, :] = x.reshape((1, A.shape[0])).copy()
        u0 = out[i, :].reshape((in_dim, 1))
    if x_out:
        return out, states
    else:
        return out