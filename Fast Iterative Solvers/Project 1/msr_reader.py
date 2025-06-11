import numpy as np
from helpers import *

def string_to_numbers_array(s, dtype="float", sep=" "):
    if s[-1:] == "\n":
        s = s[:-1]
    str_arr = s.split(sep)
    str_arr = [num for num in str_arr if num != ""]

    if dtype == "float":
        num_arr = np.array([float(string.strip()) for string in str_arr], dtype="float")
    if dtype == "int":
        num_arr = np.array([int(string.strip()) for string in str_arr], dtype="int32")
        
    return num_arr

def msr_reader(path):

    # READ FILE
    bindx = []
    values = []
    with open(path) as f:
        lines = f.readlines()
        n, nnz = string_to_numbers_array(lines[1], dtype="int") 
        bindx = np.zeros(nnz, dtype="int32")
        values = np.zeros(nnz, dtype="float")
        for i, l in enumerate(lines):
            if i == 0:
                symmetric = True if (l[0] == "s") else False
            elif i > 1:
                b, v = string_to_numbers_array(l)
                bindx[i-2] = b
                values[i-2] = v

    # CONSTRUCT MATRIX
    A = np.zeros(n*n).reshape(n,n)
    # diagonal
    for i in range(n):
        A[i,i] = values[i]
    # off-diagonal
    off_counter = 0
    for row in range(n):
        noff = int(bindx[row+1] - bindx[row])
        for i in range(noff):
            col = bindx[(n+1) + off_counter]
            val = values[(n+1) + off_counter]
            A[row][col-1] = val
            off_counter += 1
    # symmetry
    if symmetric:
        for i in range(n):
            for j in range(i):
                A[j, i] = A[i, j]
    
    return A