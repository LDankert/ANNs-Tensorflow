"""
The model class

Created: 11.12.21, 23:52

Author: LDankert
"""

from lstm_cell import LSTM_Cell
from lstm_layer import LSTM_Layer
from output_layer import Output_layer
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf


class LSTM_Model(Model):

    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.layer = LSTM_Layer(LSTM_Cell(32))
        self.outputs = Dense(5, activation= (lambda x: tf.round(tf.nn.sigmoid(x))))

    @tf.function
    def call(self,x):
        states = self.layer.zero_states(x.shape[0])
        x = self.layer(x, states)
        print(x)
        x = self.outputs(x)
        return x

