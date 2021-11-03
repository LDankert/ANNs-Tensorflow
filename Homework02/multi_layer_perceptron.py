"""
Multi Layer Perceptron Class

Created: 01.11.21, 20:22

Author: LDankert
"""

import numpy as np
from perceptron import Perceptron
from main import sigmoid,sigmoidprime

class MultiLayerPerceptron():

    # Constructor, need a list of perceptrons
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
        self.outputs = Perceptron(len(perceptrons))


    # Inputs passed through the networks
    def forward_step(self, inputs):
        perceptron = self.perceptron_generator()
        outputs = []
        for input in inputs:
            actualPerceptron = next(perceptron)
            actualPerceptron.forward_step(input)
            outputs.append(actualPerceptron.activation)
        self.outputs.forward_step(outputs)

    # Update the network parameters
    def backprob_step(self, groundTruth):
        delta = -(groundTruth * self.outputs.activation)*sigmoidprime(self.outputs.activation)
        self.outputs.update(delta)
        delta =

    # Iterates over all perceptrons
    def perceptron_generator(self):
        for perceptron in self.perceptrons:
            yield perceptron


test = MultiLayerPerceptron([Perceptron(4), Perceptron(2), Perceptron(3)])
print(test.outputs.bias)
test.forward_step([[1,2,3,4],[2,3],[4,2,1]])
print(test.outputs.bias)
test.backprob_step()
print(test.outputs.bias)
test.forward_step([[4,2,2,1],[1,4],[1,1,1]])
print(test.outputs.bias)
test.backprob_step()
print(test.outputs.bias)

