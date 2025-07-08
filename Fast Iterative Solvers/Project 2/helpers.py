import numpy as np

def unit_vector(direction=0, size=1):
    e = np.zeros(size)
    e[direction] = 1
    return e

def back_substitution(U, b):
    n = len(b)
    x = np.zeros_like(b)
    for i in reversed(range(n)):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular")
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_upper_triangular(U, b):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    for i in reversed(range(n)):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular")
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_lower_triangular(L, b):
    n = len(b)
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        if L[i, i] == 0:
            raise ValueError("Matrix is singular")
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x