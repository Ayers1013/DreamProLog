import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from .latent import *

class VariationalAutoencoder(tf.Module):
    def __init__(self):
        self.latent = ScaledNormalLatent()
    
    def encode(self):
        pass

    def decode(self):
        pass

    def loss(self, latent, sample, decoder_dist, target):
        logp_xz = decoder_dist.log_prob(target)
        logp_z = latent.base_dist.log_prob(sample)
        logq_zx = latent.log_prob(sample)
        return -tf.reduce_mean(logp_xz + logp_z -logq_zx)
