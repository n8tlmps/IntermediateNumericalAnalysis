""" 
SVD factorization takes the form A = U S V^t
where U is an m x m orthogonal matrix,
    V is an n x n orthogonal matrix,
    and S is an m x n matrix whose only nonzero elements lie along the main diagonal.

We'll assume that m >= n, and in many important applications, m is much larger than n.
"""

import numpy as np

def svd(A):
    m,n = A.shape
    # computing A^T * A
    A_tr_A = A.T @ A
    
    # computing eigenpairs of A^T * A
    eigenvalues, eigenvectors = np.linalg.eig(A_tr_A)

    # need to arrange in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    print(f"\nThe computed eigenvalues are:\n {eigenvalues}\n")
    print(f"\nThe corresponding eigenvectors are:\n {eigenvectors}\n")

    # singular values are the positive square roots of the nonzero eigenvalues of A^T.A
    s_i = np.sqrt(eigenvalues)
    S = np.zeros((m, n)) # creates a matrix with the same size as A
    np.fill_diagonal(S, s_i) # filling in diagonal points with singular values

    print(f"\nThe singular values are:\n {s_i}\n")
    print(f"\nThe singular value matrix S is: \n{S}\n")

    # V should be n x n
    V = eigenvectors
    V_tr = V.T

    print(f"\nThe orthogonal (n x n) matrix V is:\n {V}\n")
    print(f"\nThe transpose of the matrix V is:\n {V_tr}\n")

    # U should be (m x m)
    U = np.zeros((m, m))
    for i in range(n):
        U[:, i] = (1 / s_i[i]) * A @ V[:, i]
    U_known = U[:, :n]

    # fill the rest with Gram-Schmidt
    U = gram_schmidt_extend(U_known, m)

    print(f"\nThe orthogonal (m x m) matrix U is:\n {U}\n")


def proj(u, x):
    """Project x onto u (handles zero u)."""
    uu = np.dot(u, u)
    return (np.dot(u, x) / uu) * u if uu > 0 else 0.0 * x

def gram_schmidt_orthonormalize(U):
    """
    Input: U (m x k), columns are the vectors to orthonormalize.
    Output: Q (m x k) with orthonormal columns (classical GS with projections).
    """
    Q = U.astype(float).copy()
    m, k = Q.shape
    for j in range(k):
        for i in range(j):
            Q[:, j] -= proj(Q[:, i], Q[:, j])
        nrm = np.linalg.norm(Q[:, j])
        if nrm == 0:
            raise ValueError("Dependent/zero column encountered during Gram–Schmidt.")
        Q[:, j] /= nrm
    return Q

def gram_schmidt_extend(U_known, m, rng=None):
    """
    Input: U_known (m x k) — your existing columns (not necessarily orthonormal).
    Output: U_full (m x m) — completed with additional orthonormal columns via GS.
    """
    if rng is None:
        rng = np.random.default_rng()

    Q = gram_schmidt_orthonormalize(U_known)   # clean up the known columns first
    k = Q.shape[1]
    U_full = np.zeros((m, m))
    if k > 0:
        U_full[:, :k] = Q

    for j in range(k, m):
        x = rng.standard_normal(m)
        for i in range(j):
            x -= proj(U_full[:, i], x)
        nrm = np.linalg.norm(x)
        if nrm == 0:
            raise RuntimeError("Failed to find an independent vector.")
        U_full[:, j] = x / nrm

    return U_full


if __name__ == "__main__":
    A = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0],
    [1, 1, 0]
    ])
    
    svd(A)
