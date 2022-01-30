import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from abc import ABC, abstractmethod

class Latent(ABC):
    '''
    Creates a tensorflow layer to handle latent space.
    '''

    @abstractmethod
    def sample(self):
        'Sample a latent space representation.'
        pass

    @abstractmethod
    def mode(self):
        'Returns the latent space representation with highest probability, ie. the mode.'
        pass

class NormalSpace(tf.Module):
    def __init__(self, scale_init = 1.0, trainable_scale = False, )
        self._scale_init = 1.0

    def __call__(self, logit):
        