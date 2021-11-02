"""
First modul for homework 02

Created: 01.11.21, 19:17

Author: LDankert
"""

import numpy as np

# simple sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# simple derivative of the sigmoid function
def sigmoidprime(x):
    return -np.log(x / (1-x))

# input array
inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

# gates
_and =  np.array([1,0,0,0])
_or =   np.array([1,1,1,0])
_nand = np.array([0,1,1,1])
_nor =  np.array([0,0,0,1])
_xor =  np.array([0,1,1,0])
