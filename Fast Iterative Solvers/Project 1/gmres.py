import numpy as np
import matplotlib.pyplot as plt
from helpers import back_substitution
from preconditioners import jacobi, gauss_seidel, ilu0


def getKrylov(A, V, H, j, preconditioner, ilu0=False):

    # Apply Preconditioning
    w = preconditioner(A @ V[j])

    # Perform Krylov Iteration
    for i in range(j + 1):
        H[i, j] = np.dot(V[i], w)
        w -= H[i, j] * V[i]
    H[j + 1, j] = np.linalg.norm(w)
    if H[j + 1, j] != 0:
        V[j + 1] = w / H[j + 1, j]

    return V, H

def GMRES(A, b, x0, m, tol, preconditioner=None, max_iterations=600):

    n = A.shape[0]
    x = x0.copy()

    if preconditioner:
        if preconditioner == "jacobi":
            M = jacobi(A)
            M_inv = np.linalg.inv(M)  # TEMPORARY INVERSION
            apply_preconditioner = lambda v: M_inv @ v
        elif preconditioner == "gauss_seidel":
            M = gauss_seidel(A)
            M_inv = np.linalg.inv(M)  # TEMPORARY INVERSION
            apply_preconditioner = lambda v: M_inv @ v
        elif preconditioner == "ilu0":
            L, U = ilu0(A)
            apply_preconditioner = lambda v: np.linalg.solve(U, np.linalg.solve(L, v))
        else:
            raise ValueError(f"Preconditioner '{preconditioner}' not known")
    else:
        apply_preconditioner = lambda v: v  # Identity if no preconditioning
    
    b = apply_preconditioner(b)

    # Initial residual
    r = b - A @ x
    r0_norm = np.linalg.norm(r)
    beta = r0_norm

    if r0_norm == 0:
        return x, [0.0]  # Already solved

    global_errors = [1.0]  # Initial relative residual is 1

    iteration = 0  # Global iteration counter

    while iteration < max_iterations:
        # Step 1: Initialize inner GMRES storage
        V = np.zeros((m + 1, n))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        e1 = np.zeros(m + 1)

        # Step 2: Normalize residual and initialize Krylov basis
        r = b - A @ x
        beta = np.linalg.norm(r)

        if beta / r0_norm < tol:
            break

        V[0] = r / beta
        e1[0] = beta

        # Step 3: Inner GMRES loop (cycle of length m)
        for j in range(m):
            iteration += 1

            # Arnoldi

            V, H = getKrylov(A, V, H, j, apply_preconditioner)

            # Apply previous Givens rotations
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            # Compute new Givens rotation
            denom = np.hypot(H[j, j], H[j + 1, j])
            cs[j] = H[j, j] / denom if denom != 0 else 1.0
            sn[j] = H[j + 1, j] / denom if denom != 0 else 0.0

            # Apply new Givens rotation
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            # Update RHS vector
            temp = e1[j]
            e1[j] = cs[j] * temp
            e1[j + 1] = -sn[j] * temp

            # Relative residual estimate
            current_rel_residual = abs(e1[j + 1]) / r0_norm
            global_errors.append(current_rel_residual)

            # Check convergence
            if current_rel_residual < tol:
                y = back_substitution(H[:j + 1, :j + 1], e1[:j + 1])
                x = x + V[:j + 1].T @ y
                return x, global_errors

            if iteration >= max_iterations:
                break

        # End of cycle: update solution
        y = back_substitution(H[:m, :m], e1[:m])
        x = x + V[:m].T @ y

        # Final relative residual after restart
        r = b - A @ x
        beta = np.linalg.norm(r)
        rel_res = beta / r0_norm
        global_errors.append(rel_res)

        if rel_res < tol:
            break

    return x, global_errors