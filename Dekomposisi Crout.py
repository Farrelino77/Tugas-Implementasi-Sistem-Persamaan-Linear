import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        U[j, j] = 1
        for i in range(j, n):
            sum1 = sum(L[i, k] * U[k, j] for k in range(i))
            L[i, j] = A[i, j] - sum1
        for i in range(j, n):
            sum2 = sum(L[j, k] * U[k, i] for k in range(j))
            if L[j, j] == 0:
                return "Matrix is singular, cannot perform Crout decomposition"
            U[j, i] = (A[j, i] - sum2) / L[j, j]

    return L, U

def solve_linear_system_crout(A, b):
    L, U = crout_decomposition(A)
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

print("Solusi menggunakan metode dekomposisi Crout:")
print(solve_linear_system_crout(A, b))
