"""
The layer for the lstm

Created: 11.12.21, 21:47

Author: LDankert
"""

from lstm_cell import LSTM_Cell


class LSTM_Layer:

    def __init__(self, cells):
        self.cells = cells

    def call(self, x, states):
        for batch in x:
            for sequence in batch:
                for input in sequence:
                    states = self.cells(input, states)
                    x[batch[sequence[input]]] = states[0]
        return x

    def zero_states(self, batch_size):
        return tf.zeros(batch_size), tf.zeros(batch_size)
