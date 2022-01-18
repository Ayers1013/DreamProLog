from attention import *
import tensorflow as tf
import tensorflow_probability as tfp

'''
The state autoencoder decodes a set of goals embedded into R^lxd
Our data will come in the shape of (batch, goal_max, goal_length_max, dimension)
'''

class StateNet(tf.keras.layer.Model):
    def __init__(self, N, querry, output_length, d_model, num_heads, dff, rate):
        super().__init__() 
        self.encoder = Encoder(N, querry, output_length, d_model, num_heads, dff, rate)
        self.decoder = Decoder(N, output_length, d_model, num_heads, dff, rate)
        self.dense = tf.keras.layers.Dense(d_model)
    
    def call(self, x, mask, training):
        x = tf.concat(x, axis =-2)
        x = self.dense(x)
        q = self.encoder(x, mask, training)
        y = self.decoder(q, mask, training)

        return y

