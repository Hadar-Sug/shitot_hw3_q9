import numpy as np
import numpy.linalg as alg


def gram_schmidt(cols):
    w = []
    z = [cols[0]]
    z_norms = [alg.norm(z[0])]
    for i, col in enumerate(cols[1:], start=1):
        z.append(col - (np.inner(col, z[i - 1]) / np.inner(z[i - 1], z[i - 1])) * z[i - 1])
        z_norms.append(alg.norm(z[i]))
        w_i = np.zeros_like(col) if alg.norm(z[i]) == 0 else (z[i] / alg.norm(z[i]))
        w.append(w_i)
    return w, z_norms


def build_R(w, diag_vec, cols):
    R = np.diag(diag_vec)
    m = len(cols)
    for r in range(m-1):  # rows
        for c, u in enumerate(cols[r+1:], start=r+1):  # columns
            R[r][c] = np.inner(u, w[r])
    return R


if __name__ == '__main__':
    A = np.array(
        [[3, 6, 8, 0, 4, 3, 1, 5, 4, 4],
         [4, 0, 6, 5, 1, 9, 3, 3, 3, 3],
         [5, 0, 9, 8, 0, 4, 9, 6, 6, 4],
         [0, 7, 6, 9, 2, 5, 5, 5, 3, 4],
         [2, 3, 8, 1, 2, 2, 6, 6, 6, 4],
         [5, 4, 1, 8, 1, 5, 8, 9, 5, 3],
         [0, 1, 7, 5, 3, 7, 9, 4, 0, 7],
         [2, 9, 2, 8, 3, 4, 8, 2, 2, 5],
         [6, 6, 0, 0, 4, 6, 8, 2, 7, 1],
         [4, 7, 8, 6, 4, 8, 7, 8, 2, 7],
         [7, 5, 9, 9, 5, 1, 8, 4, 3, 8],
         [2, 4, 9, 2, 9, 4, 0, 7, 0, 8],
         [2, 8, 2, 4, 2, 4, 6, 3, 5, 1],
         [2, 9, 6, 8, 2, 5, 9, 0, 0, 9],
         [1, 4, 5, 2, 2, 2, 2, 6, 9, 5]]
    )
    cols = [A[:, i] for i in range(A.shape[1])]
    Q, z_norm_vec = gram_schmidt(cols)
    R = build_R(Q, z_norm_vec, cols)
    np.set_printoptions(precision=2)
    print(R)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
