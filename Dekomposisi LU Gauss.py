import numpy as np

def lu_gauss_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            for j in range(k, n):
                U[i, j] -= factor * U[k, j]

    return L, U

def solve_linear_system_lu_gauss(A, b):
    L, U = lu_gauss_decomposition(A)
    n = len(A)
    y = np.zeros(n)
    x = np.zeros(n)

    # Solve Ly = b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# Contoh penggunaan
A = np.array([[2, 1, -1],
              [3, 2, 1],
              [1, 1, 1]])
b = np.array([8, 13, 7])

print("Solusi menggunakan metode dekomposisi LU Gauss:")
print(solve_linear_system_lu_gauss(A, b))
