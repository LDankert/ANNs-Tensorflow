"""
First modul for homework 02

Created: 01.11.21, 19:17

Author: LDankert
"""

import numpy as np
from multi_layer_perceptron import MultiLayerPerceptron

# simple sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# simple derivative of the sigmoid function
def sigmoidprime(x):
    sigm = sigmoid(x)
    return sigm * (1 - sigm)

# input array
inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

# gates
gate_and =  np.array([1,0,0,0])
gate_or =   np.array([1,1,1,0])
gate_nand = np.array([0,1,1,1])
gate_nor =  np.array([0,0,0,1])
gate_xor =  np.array([0,1,1,0])

all_gates = [gate_and, gate_or, gate_nand, gate_nor, gate_xor]

# Training Area
MLP = MultiLayerPerceptron(4)