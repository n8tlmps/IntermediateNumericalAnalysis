import numpy as np

def LUfactorization(A):
    # Initialize L and U as zero matrices of size n x n
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # For i = 1 to n do:
    for i in range(n):
        # - Set l_ii = 1 (diagonal of L)
        L[i, i] = 1

    # Perform LU factorization:
    # - For i = 1 to n do:
    for i in range(n):
        # - For j = i to n do: (Compute elements of U)
        for j in range(i, n):
            # u_ij = a_ij - Σ (k = 1 to i-1) l_ik * u_kj
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

        # - If u_ii = 0, then OUTPUT "Factorization impossible" and STOP
        if U[i][i] == 0:
            print("Factorization impossible")
            exit()

        # - For j = i+1 to n do: (Compute elements of L)
        for j in range(i+1, n):
            # l_ji = (a_ji - Σ (k = 1 to i-1) l_jk * u_ki) / u_ii
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U
    
if __name__ == "__main__":
    A = np.array([
        [1, 2, -1],
        [2, -1, 4],
        [1, 2, 3]
        ])

    L, U = LUfactorization(A)
    print("Lower Triangular Matrix L\n", L)
    print("Upper Triangular Matrix U\n", U)