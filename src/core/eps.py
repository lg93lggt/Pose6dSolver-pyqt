
import numpy as np

eps = 1E-8#np.finfo(np.float).eps 

def filter(mat):
    return np.where(np.abs(mat) < eps, 0, mat)
