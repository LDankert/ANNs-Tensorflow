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
        #self.decoder = [
            layers.Dense(49, activation="sigmoid"),
            layers.Reshape((7, 7, 1, 1)),
            layers.Conv3DTranspose(8, kernel_size=(3, 3, 3), strides=(2,2,1), activation='relu', padding='same'),
            layers.Conv3DTranspose(16, kernel_size=(3, 3, 3), strides=(2,2,1), activation='relu', padding='same'),
            layers.Conv3D(1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        #print('decoder')
        #for layer in self.decoder:
        #    x = layer(x)
        #    print(x.shape)
        x = self.decoder(x)
        return x
