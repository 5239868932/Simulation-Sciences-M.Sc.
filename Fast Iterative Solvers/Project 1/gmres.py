# This is a GMRES solver

import numpy as np
import matplotlib.pyplot as plt

# def getKrylov
def unit_vector(direction=0, size=2):
    e = np.zeros(size)
    e[direction] = 1
    return e

def gmres(A, x0, b, m, tol):
    """
    A: 2D numpy array - square
    x0: 1D numpy array - initial guess
    b: 1D numpy array - right hand side
    m: integer - numer of iterations
    tol: float - tolerance
    """
    N = np.size(A, 0) # size of the square matrix (NxN)
    residuals = np.zeros(m+1, N)
    krylov_vectors = np.zeros(m+1, N)
    # e1 = unit_vector()
    print("=============")
    print(residuals)
    print("=============")
    # r0 = b - A.dot(x0)
    # v1 = 
    return None

# Initialize restarted GMRES
N = 3
A = np.random.rand(N, N)
x_sol = np.ones(N)
b = A.dot(x_sol)
x_init = np.zeros(N)
A_inv = np.linalg.inv(A)

print("Hello")