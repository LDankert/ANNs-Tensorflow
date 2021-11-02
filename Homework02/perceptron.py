"""
perceptron class

Created: 01.11.21, 20:04

Author: LDankert
"""
import numpy as np
from main import sigmoid, sigmoidprime

class Perceptron():

    #Constructor
    def __init__(self, input_units):
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn()
        self.alpha = 1

    # Calculates the activation of the perceptron
    def foward_step(self, inputs):
        activation = 0
        for input in inputs:
            activation += input
        return sigmoid(activation)

    # Updates the parameters of the perceptron
    def update(self, delta):
        for weight in self.weights:
            errorLoss = 0
        return errosLoss


test = Perceptron(4)
print(test.weights)
print(test.bias)
