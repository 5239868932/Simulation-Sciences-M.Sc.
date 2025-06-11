import numpy as np

def arnoldi(A, b, x0, m):

    # Create empty matrices and arrays
    N = A.shape[0]
    H = np.zeros((m+1, m)) # Upper Hessenberg matrix
    V = np.zeros((N, m+1)) # Orthogonal basis vectors

    r0 = b - A.dot(x0) # Initial residual vector
    V[:, 0] = r0 / np.linalg.norm(r0) # First normalized Krylov vector
    
    for j in range(m):
        w = A.dot(V[:, j]) # w: new vector to be orthogonalized
        for i in range(j + 1): # # Orthogonalize w against all previous v_i
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i] # Subtract the projection of w onto v_i
        H[j+1, j] = np.linalg.norm(w) # norm of the orthogonalized vector
        V[:, j+1] = w / H[j+1, j] # add new normalized orthogonal vector to our basis.
    
    return V, H