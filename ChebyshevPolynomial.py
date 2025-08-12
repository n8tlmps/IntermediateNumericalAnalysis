import sympy as sp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

# --------------------------------------------
# Part A: Verify Chebyshev ODE for n = 0, 1, 2, 3
# --------------------------------------------

x = sp.Symbol('x')

# Define Chebyshev polynomials symbolically
chebyshev_syms = [
    1,                  # T_0(x)
    x,                  # T_1(x)
    2*x**2 - 1,         # T_2(x)
    4*x**3 - 3*x        # T_3(x)
]

print("=== Part A: Verifying Chebyshev Differential Equation ===\n")
for n, Tn_expr in enumerate(chebyshev_syms):
    y = Tn_expr
    y_prime = sp.diff(y, x)
    y_double_prime = sp.diff(y_prime, x)
    lhs = (1 - x**2) * y_double_prime - x * y_prime + n**2 * y
    simplified_lhs = sp.simplify(lhs)

    print(f"T_{n}(x) = {y}")
    print(f"LHS = {simplified_lhs}")
    print("✅ Verified!\n" if simplified_lhs == 0 else "❌ Not satisfied.\n")


# --------------------------------------------
# Part B: Compare T_n(x) to Determinant for n = 1, 2, 3
# --------------------------------------------

def chebyshev_tridiagonal_determinant(n, x_val):
    """
    Returns the determinant of the tridiagonal matrix:
    - Row 0 diagonal is x_val
    - Rows 1 to n-1 diagonal is 2*x_val
    - All off-diagonals are 1
    """
    if n == 0:
        return 1
    elif n == 1:
        return x_val

    A = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            A[i, i] = x_val
        else:
            A[i, i] = 2 * x_val
        if i > 0:
            A[i, i-1] = A[i-1, i] = 1

    return np.linalg.det(A)

x_val = 0.5
print(f"=== Part B: Comparing T_n({x_val}) to Tridiagonal Determinants ===\n")

for n in range(1, 4):  # Only n = 1, 2, 3
    Tn = Chebyshev.basis(n)(x_val)
    det = chebyshev_tridiagonal_determinant(n, x_val)
    diff = abs(Tn - det)

    print(f"n = {n}:")
    print(f"  T_n({x_val})         = {Tn:.5f}")
    print(f"  Determinant          = {det:.5f}")
    print(f"  Absolute Difference  = {diff:.2e}\n")
