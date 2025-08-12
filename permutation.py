import numpy as np

def matrix_permutation(A):
    m, n = A.shape

    diagonal_array = np.eye(m)
    print("\nDiagonal array\n", diagonal_array)

    product = diagonal_array @ A
    print("\nproduct of A * I\n", product)

    # Create permutation matrix by swapping rows of identity
    permutation_matrix = diagonal_array.copy()
    permutation_matrix[[0, 1]] = permutation_matrix[[1, 0]]
    print("\nPermutation matrix\n", permutation_matrix)
    
    product = permutation_matrix @ A
    print("\nproduct of A * P\n", product)

if __name__ == "__main__":
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    matrix_permutation(A)