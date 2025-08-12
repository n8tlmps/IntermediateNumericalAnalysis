import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy.polynomial.polynomial import Polynomial

# Weight function w(x) = e^(-x)
def w(x):
    return np.exp(-x)

# Weighted inner product: <f, g> = ∫₀^∞ f(x)g(x)e^(-x) dx
def inner_product(f, g):
    integrand = lambda x: f(x) * g(x) * w(x)
    result, _ = quad(integrand, 0, np.inf)
    return result

# Gram-Schmidt orthogonalization to generate Laguerre polynomials
def gram_schmidt_laguerre(n):
    phi = [Polynomial([0]*i + [1]) for i in range(n + 1)]  # 1, x, x^2, ..., x^n
    L = [] # orthogonal polynomials

    for i in range(n + 1):
        phi_i = phi[i]
        for j in range(i):
            L_j = L[j]
            num = inner_product(phi_i, L_j)
            den = inner_product(L_j, L_j)
            if den == 0:
                continue
            proj = (num / den) * L_j
            phi_i = phi_i - proj
        L.append(phi_i)

    return L

# Main block to generate and plot Laguerre polynomials
if __name__ == "__main__":
    degree = 3  # Generate up to L_3(x)
    laguerre_polys = gram_schmidt_laguerre(degree)

    # Print each polynomial
    print("Laguerre Polynomials (L₀ to L₃):")
    for i, L in enumerate(laguerre_polys):
        print(f"L_{i}(x) =", L)

    # Plotting
    x_vals = np.linspace(0, 10, 400)
    plt.figure(figsize=(10, 6))

    for i, L in enumerate(laguerre_polys):
        y_vals = L(x_vals)
        plt.plot(x_vals, y_vals, label=f"L_{i}(x)")

    plt.title("Laguerre Polynomials L₀ to L₃ via Gram-Schmidt")
    plt.xlabel("x")
    plt.ylabel("L_n(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
