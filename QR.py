import numpy as np
import numpy.linalg as alg


def gram_schmidt(mat):
    cols = [mat[:, i] for i in range(A.shape[1])]
    w = np.zeros_like(mat, dtype=float)
    z = [cols[0]]
    z_norms = [alg.norm(z[0])]
    w[:, 0] = z[0] / z_norms[0]
    for i, col in enumerate(cols[1:], start=1):
        z.append(col - (np.inner(col, z[i - 1]) / np.inner(z[i - 1], z[i - 1])) * z[i - 1])
        z_norms.append(alg.norm(z[i]))
        w_i = np.zeros_like(col) if alg.norm(z[i]) == 0 else (z[i] / alg.norm(z[i]))
        w[:, i] = w_i
    return w, z_norms, cols


def build_R(w, diag_vec, cols):
    R = np.diag(diag_vec)
    m = len(cols)
    for r in range(m - 1):  # rows
        for c, u in enumerate(cols[r + 1:], start=r + 1):  # columns
            R[r][c] = np.inner(u, w[:, r])
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
    # Q1
    print("Question 1")

    Q, z_norm_vec, cols = gram_schmidt(A)
    R = build_R(Q, z_norm_vec, cols)
    np.set_printoptions(precision=2)
    print("Q")
    print(Q)
    print("R")
    print(R)

    # Q2
    print("Question 2")

    P = np.array(cols).T
    print("P")
    print(P, P.shape)

    projection_mat = np.matmul(P, P.T)
    print("projection_mat")
    print(projection_mat, projection_mat.shape)
    x = np.array([[21, 11, 9, 6, 5, 4, 2, 1, 94, 91, 89, 85, 84, 16, 98]]).T
    print("X")
    print(x, x.shape)
    x_star = np.matmul(projection_mat, x)
    print("x_star")
    print(x_star, x_star.shape)

    # Q3
    print("Question 3")

    PPT_dim = projection_mat.shape[0]
    projection_mat_orthogonal = np.identity(PPT_dim) - projection_mat
    print("projection_mat_orthogonal")
    print(projection_mat_orthogonal, projection_mat_orthogonal.shape)
    x_star_orthogonal = np.matmul(projection_mat_orthogonal, x)
    print("x_star_orthogonal")
    print(x_star_orthogonal, x_star_orthogonal.shape)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
