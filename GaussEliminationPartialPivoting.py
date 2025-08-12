import numpy as np

def GaussEliminationPartialPivoting(A, b):
    n = len(b)
    x = np.zeros(n)
    new_line = "\n"

    # Combine A and b into augmented matrix
    augmented_matrix = np.hstack((A, b))
    print(f"The initial augmented matrix is:{new_line}{augmented_matrix}{new_line}")
    print("Solving for the upper-triangular matrix...", new_line)

    # Gaussian Elimination with Partial Pivoting
    for i in range(n):
        # Partial Pivoting
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        if max_row != i:
            augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
            print(f"Swapped row {i} with row {max_row}:{new_line}{augmented_matrix}{new_line}")

        # Elimination
        for p in range(i + 1, n):
            if augmented_matrix[i, i] == 0:
                continue
            factor = augmented_matrix[p, i] / augmented_matrix[i, i]
            augmented_matrix[p, i:] -= factor * augmented_matrix[i, i:]
            print(f"Eliminated row {p} using row {i}:{new_line}{augmented_matrix}{new_line}")

    print(f"Upper triangular matrix:{new_line}{augmented_matrix}{new_line}")

    # Back Substitution
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n])) / augmented_matrix[i, i]

    print(f"Solution vector x:{new_line}{x}{new_line}")
    return x

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    A = np.array([[3, 2, -4],
                  [2, 3, 3],
                  [5, -3, 1]], dtype=float)
    b = np.array([[3],
                  [15],
                  [14]], dtype=float)

    GaussEliminationPartialPivoting(A, b)
