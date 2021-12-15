"""
The combining autoencoder class

Created: 14.12.21, 22:20

Author: LDankert
"""
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from tensorflow.keras import Model


class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x