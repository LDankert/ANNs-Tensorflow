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
        self.outputs = Perceptron(len(self.perceptrons))


    # Inputs passed through the networks
    def forward_step(self, inputs):
        perceptron = self.perceptron_generator()
        activations = []
        for input in inputs:
            actualPerceptron = next(perceptron)
            actualPerceptron.forward_step(input)
            activations.append(actualPerceptron.activation)
        self.outputs.forward_step(activations)

    # Update the network parameters
    def backprob_step(self, groundTruth):
        # First calc the output depending of the ground truth and update the output perceptron
        deltaOutput = -(groundTruth * self.outputs.activation)*sigmoidprime(self.outputs.activation)
        self.outputs.update(deltaOutput)
        # Now update all perceptrons with delta depending on first delta
        for perceptron in self.perceptrons:
            delta = sum(deltaOutput * perceptron.weights) * sigmoidprime(perceptron.activation)
            perceptron.update(delta)

    # Output perceptron setter weigths has to match perceptrons
    def set_outputs(self,perceptron):
        if len(perceptron.weights) != len(self.perceptrons):
            raise Exception ("Weights of output perceptron does not match number of perceptrons")
        else:
            self.outputs = perceptron

    # Perceptron setter, needs a list of Perceptrons, generates a fitting output perceptron too
    def set_perceptrons(self,perceptrons):
        self.perceptrons = perceptrons
        self.outputs = Perceptron(len(self.perceptrons))

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

