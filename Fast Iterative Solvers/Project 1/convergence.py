import os
import matplotlib.pyplot as plt
import numpy as np
from msr_reader import msr_reader
from gmres import GMRES

# Read matrix data
path = os.getcwd()
matrices = ["cg_matrix_msr_1.txt", # 0
            "cg_matrix_msr_2.txt", # 1
            "gmres_matrix_msr_1.txt", # 2
            "msr_test_non_symmetric.txt", # 3
            "msr_test_symmetric.txt"] # 4

path = path + f"\\Fast Iterative Solvers\\Project 1\\matrices\\{matrices[2]}"

A = msr_reader(path)

# Setup up initial conditions
n = A.shape[0]
x = np.ones(n)
b = A.dot(x)
x0 = np.zeros(n)
m = 600
tol = 1e-8

# Quick Check, if on the diagonal there are any zeros!
# A_diagonal_entries = np.diag(A)
# number_of_zeros = n - np.count_nonzero(A_diagonal_entries)
# print("Zeros:", number_of_zeros)

# Apply GMRES
x_approx, errors = GMRES(
    A,
    b,
    x0,
    m,
    tol,
    preconditioner="ilu0",
    max_iterations=600)


# Plot convergence
iterations = range(len(errors))

plt.figure(figsize=(6, 4))
plt.plot(iterations, np.abs(errors))
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm (log scale)')
plt.title('Convergence of GMRES')
plt.grid(True)

plt.show()