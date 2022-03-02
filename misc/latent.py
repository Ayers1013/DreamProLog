import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class LatentSpace:
    '''
    Creates a tensorflow layer to handle latent space. 
    Subclasses may have self._dist, otherwise sample, mode methods have to be overwritten.
    '''

    def sample(self, *args, **kwargs):
        return self._dist.sample(*args, **kwargs)

    def mode(self, *args, **kwargs):
        return self._dist.mode(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self._dist.log_prob(*args, **kwargs)
    
    def extract(self, training):
        if training: return self.sample()
        else: return self.mode()

    def compare(self, other):
        return tf.reduce_sum(self.comparison_loss(other))/(512*16*128)

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
    def __init__(self, d_model, scale_min=0.001):
        super().__init__()
        self._scale_min = scale_min

        self._latent_dense = tf.keras.layers.Dense(d_model)
        self._scale_dense = tf.keras.layers.Dense(d_model, activation = 'sigmoid')
        
        self.base_dist = tfd.Independent(
            tfd.Normal(tf.zeros(d_model), 1.),
            reinterpreted_batch_ndims = 1
        )

    def __call__(self, logit):
        self._loc = self._latent_dense(logit)
        self._scale = self._scale_dense(logit)
        if self._scale_min: self._scale = tf.math.minimum(self._scale, self._scale_min)
        self._dist = tfd.Independent(
            tfd.Normal(self._loc, self._scale),
            reinterpreted_batch_ndims = 1)
        return self

    def sample(self):
        scale = tf.stop_gradient(self._scale)
        dist = tfd.Independent(
            tfd.Normal(self._loc, scale),
            reinterpreted_batch_ndims = 1)
        return dist.sample()
    
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
        probs = tf.math.softmax(logits, axis = -1)
        self._dist = tfd.OneHotCategorical(probs = probs)
        return self

    def sample(self):
        x = tf.cast(self._dist.sample(), tf.float32) + self._dist.probs - tf.stop_gradient(self._dist.probs)
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
        self._logits = self._dense_1(logits)
        self._dist = tfd.RelaxedOneHotCategorical(self._temp, self._logits)
        return self

    def sample(self):
        x = self._dist.sample()
        x = self._dense_2(x)
        return x

    def mode(self):
        x = tfd.RelaxedOneHotCategorical(1e-5, self._logits).sample()
        x = self._dense_2(x)
        return x

    def comparison_loss(self, latent):
        assert isinstance(latent, GumbleSpace)
        p, q = tf.math.softmax(self._logits, axis = -1), tf.math.softmax(latent._logits, axis = -1)
        return tf.reduce_sum(p*(tf.math.log(p)-tf.math.log(q)), axis=-1)

