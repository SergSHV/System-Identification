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
    i = u.shape[0] - 1
    while len(true_list) != r:
        new_r = np.linalg.matrix_rank(u[list(true_list.difference({i})), :])
        if new_r == r:
            true_list = true_list.difference({i})
        i -= 1
    return list(true_list)


def generate_independent_vectors(matrix, num_edges):
    step = 0
    n = matrix.shape[0]
    r = 0
    for i in range(n):
        if i == 0:
            if sum(matrix[i, -num_edges:]) == 0:
                matrix[i, step] = 1
                step += 1
            r = 1
        else:
            if i - step == num_edges:  # all independent vectors are found
                matrix[i, step] = 1
                step += 1
            elif step != n - num_edges:  # no all dependent vectors found
                new_r = np.linalg.matrix_rank(matrix[:(i + 1), -num_edges:])
                if new_r == r:
                    matrix[i, step] = 1
                    step += 1
                else:
                    r += 1
    return matrix


def gen_vectors(u, per):
    inp_dim, ind_list = define_dim(u, getlist=True)
    ind_list = np.delete(range(u.shape[1]), ind_list)

    matrix = np.zeros((inp_dim * per, per))

    for d in range(per):
        for i in range(u.shape[0]):  # create Matrix
            matrix[d * inp_dim:(d + 1) * inp_dim, (i - d) % (matrix.shape[1])] = u[i, ind_list].T.copy()

    r = np.linalg.matrix_rank(matrix)

    if r == matrix.shape[1]:
        dim = matrix.shape[1]
        basic_var = list(range(dim))
    else:
        matrix = matrix[~np.all(matrix == 0, axis=1)]
        matrix = matrix[:binary_search(matrix, r, matrix.shape[0], r), :]
        #  matrix = matrix[remove_dependant_elements(matrix, r), :]
        basic_var = remove_dependant_rows(matrix.T, r)  # remove_dependant_elements(A.T,r)
        dim = len(basic_var)
    vec = np.zeros((per, dim))
    vec[:, -u.shape[1]:] = u[:per, :]
    vec[basic_var, :] = generate_independent_vectors(vec[basic_var, :], u.shape[1])

    if r != matrix.shape[1]:
        for i in range(per):
            if i not in basic_var:
                roots = np.linalg.lstsq(matrix[:, basic_var], matrix[:, i], rcond=None)[0]
                x = np.zeros((1, dim - u.shape[1]))
                for v in basic_var:
                    for j in range(dim - u.shape[1]):
                        x[0, j] += vec[v, j] * roots[v]
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


def lpg_gen(u, y):
    """
    :param u: input vectors
    :param y: output vectors
    :return: system matrix q, initial state vector x0
    """
    q, x = identify_system(u)
    return q, x[:-u.shape[1]]
