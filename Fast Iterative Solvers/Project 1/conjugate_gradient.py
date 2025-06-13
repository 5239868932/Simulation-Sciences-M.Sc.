import numpy as np

def conjugate_gradient(A, b, x0, m, tol):
    
    # scalars
    n = A.shape[0]
    # lists of scalars
    x_approx = np.zeros(n)
    alpha = np.zeros(m+1)
    beta = np.zeros(m+1)
    a_norm_errors = np.zeros(m)
    # lists of vectors
    residual = np.zeros((m+1, n))
    p = np.zeros((m+1, n))

    # Initial Residual
    residual[0] = b - A @ x0
    p[0] = residual[0].copy()

    # Main Loop
    for i in range(m):
        # precompute
        Apm = A @ p[i]

        alpha[i] = residual[i].T.dot(residual[i]) / p[i].T.dot(Apm)
        x_approx = x_approx + alpha[i] * p[i]
        residual[i+1] = residual[i] - alpha[i] * Apm
        beta[i+1] = residual[i+1].T.dot(residual[i+1]) / residual[i].T.dot(residual[i])
        p[i+1] = residual[i+1] + beta[i] * p[i]

        # A-norm errors
        error = x_approx - 1 # true x is a vector with only ones
        a_norm = lambda A, e: np.sqrt(e.T.dot(A@e))
        a_norm_errors[i] = a_norm(A, error)

    return x_approx, a_norm_errors, residual