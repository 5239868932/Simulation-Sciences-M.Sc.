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
    A = A.copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = (A != 0)

    for i in range(1, n):
        for j in range(i):
            if P[i, j] and U[j, j] != 0:
                L[i, j] = U[i, j] / U[j, j]
                for k in range(j, n):
                    if P[i, k]:
                        U[i, k] -= L[i, j] * U[j, k]

    U[~P] = 0
    return L, U


if __name__ == "__main__":
    A = np.array([[1, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]])
    ilu0(A)