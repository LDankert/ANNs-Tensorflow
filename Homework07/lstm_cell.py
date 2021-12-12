"""
LSTM cell Class

Created: 11.12.21, 19:04

Author: LDankert
"""
import tensorflow as tf
from tensorflow.keras import layers


class LSTM_Cell:

    # initialize 4 gates: forget
    def __init__(self, units):
        # forget gate with activation function sigmoid and 1 as bias
        self.forgetGate = layers.Dense(units, activation="sigmoid", bias_initializer='Ones')
        # input gate with activation function sigmoid and 1 as bias
        self.inputGate = layers.Dense(units, activation="sigmoid", bias_initializer='Ones')
        # output gate with activation function sigmoid and 1 as bias
        self.outputGate = layers.Dense(units, activation="sigmoid", bias_initializer='Ones')
        # cell state candidates gate with activation function sigmoid and 1 as bias
        self.cellStateCandidates = layers.Dense(units, activation="tanh", bias_initializer='Ones')

    # the call function, takes the input x and the states from the previous time step as tuple
    def call(self, x, states):
        # splits the tuple
        hidden_state, cell_state = states
        # concatenate the h_t-1 and the input x
        input_states = tf.concat([hidden_state, x], axis=-1)
        # calculate the next cell state from the forgetGate, inputGate
        next_cell_state = tf.matmul(self.forgetGate(input_states), cell_state) + \
                          tf.matmul(self.inputGate(input_states), self.cellStateCandidates(input_states))
        # calculate the next hidden state from the output gate and the next cell state
        next_hidden_state = tf.matmul(self.outputGate(input_states), tf.math.tanh(next_cell_state))

        return next_hidden_state, next_cell_state
