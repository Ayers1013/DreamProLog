import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class LatentSpace:
    '''
    Creates a tensorflow layer to handle latent space. 
    Subclasses may have self._dist, otherwise sample, mode methods have to be overwritten.
    '''

    def __getattr__(self, name):
        return getattr(self._dist, name)
    
    def extract(self, training):
        if training: return self.sample()
        else: return self.mode()

class NormalSpace(tf.Module, LatentSpace):
    def __init__(self, scale_init = .1):
        super().__init__()
        self._scale_init = 1.0
        self._dist = None

    def __call__(self, logit):
        self._dist = tfd.Independent(
            tfd.Normal(loc = logit, scale = self._scale_init),
            reinterpreted_batch_ndims = 1)
        return self

    def comparison_loss(self, latent):
        assert isinstance(latent, NormalSpace)
        return tfd.kl_divergence(self._dist, latent._dist)

class ScaledNormalSpace(tf.Module, LatentSpace):
    def __init__(self, d_model, scale_init = 1.0, scale_min = 0.001):
        super().__init__()
        self._latent_dense = tf.keras.layers.Dense(d_model)
        self._scale_dense = tf.keras.layers.Dense(1, activation = 'sigmoid')

    def __call__(self, logit):
        loc = self._latent_dense(logit)
        scale = self._scale_dense(logit)
        self._dist = tfd.Independent(
            tfd.Normal(self._dist.loc, scale),
            reinterpeted_batch_ndims = 1)
        return self

    def sample(self):
        scale = tf.stop_gradient(self._dist.scale)
        dist = tfd.Independent(
            tfd.Normal(self._dist.loc, scale),
            reinterpeted_batch_ndims = 1)
        dist.sample()
    
    def comparison_loss(self, latent):
        assert isinstance(latent, ScaledNormalSpace)
        return tfd.kl_divergence(self._dist, latent._dist)

class DiscrateSpace(tf.Module, LatentSpace):
    def __init__(self, categories, d_model):
        super().__init__()
        self._dense_1 = tf.keras.layers.Dense(categories)
        self._dense_2 = tf.keras.layers.Dense(d_model)

    def __call__(self, logits):
        logits = self._dense_1(logits)
        self._dist = tfd.OneHotCategorical(logits)
        return self

    def sample(self):
        x = self._dist.sample() + self._dist.probs - tf.stop_gradient(self._dist.probs)
        x = self._dense_2(x)
        return x
    
    def mode(self):
        return self._dense_2(self._dist.mode())

    def comparison_loss(self, latent):
        assert isinstance(latent, DiscrateSpace)
        return tfd.kl_divergence(self._dist, latent._dist)


class GumbleSpace(tf.Module, LatentSpace):
    def __init__(self, categories, d_model, temperature):
        super().__init__()
        self._dense_1 = tf.keras.layers.Dense(categories)
        self._dense_2 = tf.keras.layers.Dense(d_model)
        self._temp = temperature

    def __call__(self, logits):
        logits = self._dense_1(logits)
        self._dist = tfd.RelaxedOneHotCategorical(self._temp, logits)
        return self

    def sample(self):
        x = self._dist.sample()
        x = self._dense_2(x)
        return x

    def mode(self):
        return self._dense_2(self._dist.mode())

    def comparison_loss(self, latent):
        assert isinstance(latent, GumbleSpace)
        p, q = self._dist.probs, latent._dist.probs
        return tf.reduce_sum(p*(tf.math.log(p)-tf.math.log(q)), axis=-1)

