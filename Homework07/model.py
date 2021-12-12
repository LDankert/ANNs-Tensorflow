"""
The model class

Created: 11.12.21, 23:52

Author: LDankert
"""

from lstm_cell  import LSTM_Cell
from lstm_layer import  LSTM_Layer
from tensorflow.keras import Model


class Model:

    def __init__(self):
        #super(Model, self).__init__()
        self.layers = [
            readinlayer
            LSTM_Layer(LSTM_Cell(5))
            outputlayer
        ]

    def call(self,x):
        x = self.layer(x)
        return x

