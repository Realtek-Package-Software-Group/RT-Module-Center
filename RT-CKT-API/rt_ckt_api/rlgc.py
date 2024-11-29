import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import numba as nb

from rt_math_api.utility import ExpressionValue, UNIT_TO_VALUE


#%% Math Function


def maxwell_to_spice(matrix_type: str, matrix: np.ndarray) -> np.ndarray:
    
    if matrix_type.lower() not in ['r', 'l', 'g', 'c']:
        raise ValueError('Invalid matrix type')
    
    if matrix_type.lower() in ['r', 'l']:
        spice_matrix = matrix
    else: #* C G
        dim = matrix.shape[0]
        spice_matrix = np.zeros_like(matrix)
        for i in range(dim):
            for j in range(dim):
                spice_matrix[i, j] = -matrix[i, j] if i != j else matrix[i,i] + sum([matrix[i, k] for k in range(dim) if k != i])
    
    return spice_matrix
                
def float_at_infinity(matrix_type: str, matrix: np.ndarray) -> np.ndarray:
    
    if matrix_type.lower() not in ['r', 'l', 'g', 'c']:
        raise ValueError('Invalid matrix type')
    

    if matrix_type.lower() in ['r', 'l']:
        new_matrix = matrix
    
    else: #* C G

        dim = matrix.shape[0]
        new_matrix = np.zeros((dim, dim))   
        
        #* Assume the given matrix is a Maxwell matrix
        spice_matrix = maxwell_to_spice(matrix_type, matrix)
        sum_self_c = sum([spice_matrix[k, k] for k in range(dim)]) 
        for i in range(dim):
            for j in range(dim):
                
                c_ij = spice_matrix[i, j]
                c_ii = spice_matrix[i, i]
                c_jj = spice_matrix[j, j]
                
                if i != j:
                    new_matrix[i, j] = -(c_ij + c_ii*c_jj/sum_self_c)
                else:
                    new_matrix[i, j] = c_ii*sum([spice_matrix[k, k] for k in range(dim) if k != i])/sum_self_c + sum(spice_matrix[i, k] for k in range(dim) if k != i)
                
    return new_matrix


def float_net(matrix_type: str, matrix: np.ndarray, floating_index: int) -> np.ndarray:
    
    if matrix_type.lower() not in ['r', 'l', 'g', 'c']:
        raise ValueError('Invalid matrix type')
    
    dim = matrix.shape[0]
    if not -dim <= floating_index < dim:
        raise ValueError('Invalid floating index')
    
    floating_index = floating_index % dim
    new_matrix = np.zeros((dim-1, dim-1))

    if matrix_type.lower() in ['r', 'l']:

        for i in range(dim):
            
            new_i = i if i < floating_index else i-1
            if i == floating_index:
                continue
            
            for j in range(dim):
                
                new_j = j if j < floating_index else j-1
                if j == floating_index:
                    continue
                
                new_matrix[new_i, new_j] = matrix[i,j]
        
    else: #* C G
        '''To the floating net and the ground net, sequentially.'''        
        for i in range(dim):
            
            new_i = i if i < floating_index else i-1
            if i == floating_index:
                continue
            
            for j in range(dim):
                
                new_j = j if j < floating_index else j-1
                if j == floating_index:
                    continue
                
                c_ij = matrix[i, j]
                c_ik = matrix[i, floating_index]
                c_kj = matrix[floating_index, j]
                c_kk = matrix[floating_index, floating_index]
                
                new_matrix[new_i, new_j] = c_ij - c_ik*c_kj/c_kk
    
    return new_matrix


def ground_net(matrix_type: str, matrix: np.ndarray, grounded_index: int) -> np.ndarray:
    
    if matrix_type.lower() not in ['r', 'l', 'g', 'c']:
        raise ValueError('Invalid matrix type')
    
    dim = matrix.shape[0]
    if not -dim <= grounded_index < dim:
        raise ValueError('Invalid floating index')
    
    floating_index = grounded_index % dim
    new_matrix = np.zeros((dim-1, dim-1))


    if matrix_type.lower() in ['r', 'l']:

        for i in range(dim):
            
            new_i = i if i < grounded_index else i-1
            if i == grounded_index:
                continue
            
            for j in range(dim):
                
                new_j = j if j < grounded_index else j-1
                if j == grounded_index:
                    continue
                
                l_ij = matrix[i, j]
                l_kj = matrix[grounded_index, j]
                l_ik = matrix[i, grounded_index]
                l_kk = matrix[grounded_index, grounded_index]
                
                new_matrix[new_i, new_j] = l_ij - l_ik*l_kj/l_kk
        
    else: #* C G
        '''To the floating net and the ground net, sequentially.'''
        for i in range(dim):
            
            new_i = i if i < grounded_index else i-1
            if i == grounded_index:
                continue
            
            for j in range(dim):
                
                new_j = j if j < grounded_index else j-1
                if j == grounded_index:
                    continue
            
            new_matrix[new_i, new_j] = matrix[i, j]
    
    
    return new_matrix    
    
    
def return_path(matrix_type: str, matrix: np.ndarray, returned_index: int) -> np.ndarray:
    
    if matrix_type.lower() not in ['r', 'l', 'g', 'c']:
        raise ValueError('Invalid matrix type')
    

    dim = matrix.shape[0]
    shape = matrix.shape
    new_matrix = np.zeros((dim-1, dim-1))

    if matrix_type.lower() in ['r', 'l']:

        for i in range(dim):
            new_i = i if i < returned_index else i-1
            if i == returned_index:
                continue
            for j in range(dim):
                new_j = j if j < returned_index else j-1
                if j == returned_index:
                    continue
                l_ij = matrix[i, j]
                l_kj = matrix[returned_index, j]
                l_ik = matrix[i, returned_index]
                l_kk = matrix[returned_index, returned_index]
                
                new_matrix[new_i, new_j] = l_ij - l_kj - l_ik + l_kk
        
    elif  matrix_type.lower() in ['g', 'c']:
        
        float_at_infinity_matrix = float_at_infinity(matrix_type, matrix)
        new_matrix = ground_net(matrix_type, float_at_infinity_matrix, returned_index)
    
    
    return new_matrix

    



#%% Object

class RMatrix:
    matrix_type = 'R'
    matrix_format = 'Maxwell'
    def __init__(self):
        ... 
    
class LMatrix:
    matrix_type = 'L'
    matrix_format = 'Maxwell'
    def __init__(self):
        ... 
        
class GMatrix:
    matrix_type = 'G'
    matrix_format = 'Maxwell'
    def __init__(self):
        ...     
        
class CMatrix:
    '''Assume to be Maxwell matrix'''
    matrix_type = 'C'
    matrix_format = 'Maxwell'
    def __init__(self):
        ... 


#%% Test

if __name__ == '__main__':
    
    # c_matrix = np.array([
    #     [1.41425, -1.24703, -0.05650],
    #     [-1.24703, 2.52679, -1.24718],
    #     [-0.05650, -1.24718, 1.41453]
    #     ])
    
    # l_matrix = np.array([
    #     [1.58802, 1.44881, 1.34183],
    #     [1.44881, 1.57498, 1.44890],
    #     [1.34183, 1.44890, 1.58792]
    # ])
    
    c_matrix = np.array([
        [1.91438, -0.31573, -1.22443, -0.18926],
        [-0.31573, 3.49369, -1.07587, -2.04621],
        [-1.22443, -1.07587, 2.52529, -0.16173],
        [-0.18926, -2.04621, -0.16173, 2.54894]
    ])
    
    l_matrix =  np.array([
        [18.63865, 11.34468, 13.81999, 9.93892],
        [11.34468, 20.32677, 14.66634, 14.85084],
        [13.81999, 14.66634, 22.17347, 11.85285],
        [9.93892, 14.85084, 11.85285, 20.50591]
    ])
    
    l_matrix2 = float_net('L', l_matrix, 3)
    
    print(return_path('L', return_path('L', l_matrix2, 0), 0))
    print(ground_net('L', ground_net('L', l_matrix2, 0), 0))
    
    
    # print(float_net('C', c_matrix, 3))
    
    
    # print(maxwell_to_spice('C', c_matrix))
    
    
    # print(float_at_infinity('C', c_matrix))
    
    # print(ground_net('L', ground_net('L', l_matrix, 0), 0))
    # print(return_path('L', return_path('L', l_matrix, 0), 0))
    