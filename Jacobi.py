import numpy as np

# P_i = 0.5P_{i-1} + 0.5P_{i+1} for each i = 1, 2, ..., n-1
# P_0 = 1 and P_N = 0



def create_matrix(n):
    A = np.zeros((n-1, n-1))

    for i in range(n-1):
        A[i, i] = 1 # diagonal
        if i > 0:
            A[i, i-1] = -0.5 # left
        if i < n-2:
            A[i, i+1] = -0.5 # right

    return A

def create_matrix_alpha(n, alpha):
    A = np.zeros((n-1, n-1))

    for i in range(n-1):
        A[i, i] = 1
        if i > 0:
            A[i, i-1] = -alpha # left
        if i < n-2:
            A[i, i+1] = - (1 - alpha) # right

    return A

def create_rhs_b(n):
    b = np.zeros(n-1)
    b[0] = 0.5
    return b

def create_rhs_b_alpha(n, alpha):
    b = np.zeros(n-1)
    b[0] = alpha
    return b


def solve_jacobi(A, b, N, tol):
    x0 = np.zeros(len(b))
    n = len(b)
    x = np.zeros(n)

    k = 1
    while k <= N:
        for i in range(n):
            x[i] = (- sum(A[i, j] * x0[j] for j in range(n) if j != i) + b[i]) / A[i, i]

        if np.linalg.norm(x - x0, ord=np.inf) < tol:
            print(f"Solution converged in {k} iterations! :)")
            return x

        x0[:] = x # update x0 for the next iteration
        # print(f"{x}: i={k}") # comment in this line if you wanna see how it iterates
        k += 1

    if k > N:
        print("Maximum number of iterations exceeded.")

if __name__ == "__main__":
    print("------------------------------------------------------")
    print("part a")
    print("------------------------------------------------------")
    n = 10
    A = create_matrix(n)
    b = create_rhs_b(n)
    N = 1000
    tol = 1e-8

    print("\nA:\n", A)
    print("\nRHS b:\n", b)
    print("\nIterating solution p s.t. Ap = b with Jacobi solver...\n")
    p = solve_jacobi(A, b, N, tol)
    print(p)

    print("------------------------------------------------------")
    print("part b")
    print("------------------------------------------------------")

    A2 = create_matrix_alpha(n, 0.3)
    print("\nA with alpha=0.3:\n", A2)
    b2 = create_rhs_b_alpha(n, 0.3)
    print("\nRHS b with alpha=0.3:\n", b2)
    print("\nIterating solution p s.t. Ap = b with Jacobi solver...\n")
    p2 = solve_jacobi(A2, b2, N, tol)
    print(p2)

    print("yaayyyy")

    
    