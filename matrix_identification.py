from utilities import *


def binary_search(u, low, high, r):
    # Check base case
    if high > low:
        mid = (high + low) // 2
        # If element is present at the middle itself
        new_r = np.linalg.matrix_rank(u[:mid, :]) if mid != 0 else 0
        if new_r < r:
            return binary_search(u, mid + 1, high, r)
        else:
            return binary_search(u, low, mid, r)
    else:
        return high


def remove_dependant_elements(u, r):
    u_copy = u.copy()
    ind_list = []
    step = 0
    while step != r:
        min_rows = binary_search(u_copy, r - step, u_copy.shape[0], r - step)
        ind_list.append(min_rows - 1)
        u_copy = u_copy[:min_rows, :]
        step += 1
    return ind_list


def remove_dependant_rows(u, r):
    true_list = set(range(u.shape[0]))
    new_r = 0
    i = u.shape[0] - 1
    while new_r != u.shape[1]:
        new_r = np.linalg.matrix_rank(u[list(true_list.difference({i})), :])
        if new_r == r:
            true_list = true_list.difference({i})
        i -= 1
    return list(true_list)


def gen_vectors(u, per):
    inp_dim = define_dim(u)
    matrix = np.zeros((inp_dim * per, per))

    for d in range(per):
        for i in range(u.shape[0]):  # create Matrix
            matrix[d * inp_dim:(d + 1) * inp_dim, (i - d) % (matrix.shape[1])] = u[i, :].T.copy()

    r = np.linalg.matrix_rank(matrix)

    if r == matrix.shape[1]:
        basic_var = list(range(matrix.shape[1]))
    else:
        matrix = matrix[~np.all(matrix == 0, axis=1)]
        matrix = matrix[remove_dependant_elements(matrix, r), :]
        basic_var = remove_dependant_rows(matrix.T, r)  # remove_dependant_elements(A.T,r)

    dim = len(basic_var)
    vec = np.zeros((per, dim))

    vec[:per, -u.shape[1]:] = u[:per, :]
    for i in basic_var:
        vec[i, :dim - u.shape[1]] = np.random.randint(2, size=dim - u.shape[1])
        while min(vec[i, :]) == max(vec[i, :]):
            vec[i, :dim - u.shape[1]] = np.random.randint(2, size=dim - u.shape[1])
        if i != 0:
            r1, r2 = 0, 0
            while r1 == r2:
                vec[i, :dim - u.shape[1]] = np.random.randint(2, size=dim - u.shape[1])
                if min(vec[i, :]) != max(vec[i, :]):
                    r1 = np.linalg.matrix_rank(vec[:i, :])
                    r2 = np.linalg.matrix_rank(vec[:i + 1, :])

    for i in range(per):
        if i not in basic_var:
            roots = np.linalg.lstsq(matrix[:, basic_var], matrix[:, i], rcond=None)[0]
            x = np.zeros((1, dim - u.shape[1]))
            for v in basic_var:
                for j in range(dim - u.shape[1]):
                    x[0, j] = x[0, j] + vec[v, j] * roots[v]
            vec[i, :dim - u.shape[1]] = x.copy()
    return vec


def solve_sys(vec):
    dim = vec.shape[0]
    vec2 = vec.copy()

    for i in range(dim):
        if i != dim - 1:
            vec2[i, :] = vec[i + 1, :].copy()
        else:
            vec2[i, :] = vec[0, :].copy()
    x = np.linalg.lstsq(vec, vec2,rcond=None)[0].T
    return x


def identify_system(u):
    # Step 1. Periodicity
    t = define_periodicity(u)
    # Steps 2-3. Solve the system and generate vectors
    vectors = gen_vectors(u[:t, :], t)
    # Step 4. Find Q
    q = solve_sys(vectors)
    return q, vectors[0, :]


def simulate_sequence(u, q, x):
    v = x.T.copy()
    out = np.zeros(u.shape)
    for i in range(u.shape[0]):
        v = q @ v
        out[i, :] = v[-u.shape[1]:].copy()
    return out
