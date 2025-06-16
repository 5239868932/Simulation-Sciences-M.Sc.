import numpy as np

def conjugate_gradient(A, b, x0, m, tol):
    
    # scalars
    n = A.shape[0]
    # lists of scalars
    alpha = np.zeros(m)
    beta = np.zeros(m)
    a_norm_errors = np.zeros(m)
    rel_residuals = np.zeros(m)
    # lists of vectors
    x_approx = np.zeros((m+1, n))
    residual = np.zeros((m+1, n))
    p = np.zeros((m+1, n))

    # Initialization
    x_approx[0] = x0
    residual[0] = b - A @ x_approx[0]
    p[0] = residual[0].copy()

    # Precompute
    rel_res_0 = np.linalg.norm(residual[0])

    # Main Loop
    for k in range(m):
        # precompute matrix-vector product
        Apm = A @ p[k]
        # find from x_k in direction p_(k+1) the new location
        # x_(k+1) of mimimum of A and update the gradient /residual
        alpha[k] = residual[k].T.dot(residual[k]) / p[k].T.dot(Apm)
        x_approx[k+1] = x_approx[k] + alpha[k] * p[k]
        residual[k+1] = residual[k] - alpha[k] * Apm
        # correct the search direction p_(k+1) using p_k and r_(k+1)
        beta[k] = residual[k+1].T.dot(residual[k+1]) / residual[k].T.dot(residual[k])
        p[k+1] = residual[k+1] + beta[k] * p[k]  # Use correct index

        # Definition of A-norm 
        a_norm = lambda A, e: np.sqrt(e.T.dot(A@e))

        # compute A-norm errors
        error = x_approx[k+1] - np.ones(n) # true x is a vector with only ones
        a_norm_errors[k] = a_norm(A, error)

        # Relative residuals
        rel_residuals[k] = np.linalg.norm(residual[k]) / rel_res_0
        if rel_residuals[k] < tol:
            break

    return x_approx[k+1], a_norm_errors[:k], residual[1:k+1], rel_residuals[:k]


if __name__ == "__main__":

    # TEST SCRIPT

    import numpy as np
    from time import process_time

    n = 1000
    a = np.random.randint(2, size=(n,n)) + np.identity(n, dtype="int8")
    A = a.T.dot(a)

    x_exact = np.ones(n)
    b = A.dot(x_exact)
    x0 = np.zeros(n)
    tol = 1e-8
    m = 2000

    s = process_time()

    x_approx, a_norm_errors, residual, rel_residuals = conjugate_gradient(A, b, x0, m, tol)

    e = process_time()
    print(f"Elapsed time: {round(e-s,6)} seconds")

    for res in rel_residuals[-5:]:
        print(x_approx[-10:])
