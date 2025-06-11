import numpy as np

def jacobi(A):
    n = A.shape[0]
    M = np.identity(n) * np.diag(A)
    return M

def gauss_seidel(A):
    n = A.shape[0]
    M_vector = A[np.tril_indices(n, k=0)]
    M = np.zeros((n, n))
    M[np.tril_indices(n, k=0)] = M_vector
    return M

def ilu0(A):
    return None