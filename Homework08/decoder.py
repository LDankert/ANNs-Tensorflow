"""
The decoder class

Created: 14.12.21, 22:15

Author: LDankert
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        x = self.decoder(x)
        return x
