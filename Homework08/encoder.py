"""
The encoder class

Created: 14.12.21, 22:11

Author: LDankert
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    def call(self, x):
        x = self.encoder(x)
        return x
