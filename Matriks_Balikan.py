import numpy as np

def solve_linear_system_inverse(A, b):
    if np.linalg.det(A) == 0:
        return "Matrix is singular, cannot find inverse"
    
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x

# Contoh penggunaan
A = np.array([[2, 1, -1],
              [3, 2, 1],
              [1, 1, 1]])
b = np.array([8, 13, 7])

print("Solusi menggunakan metode matriks balikan:")
print(solve_linear_system_inverse(A, b))
