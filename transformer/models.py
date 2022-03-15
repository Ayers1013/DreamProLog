import tensorflow as tf
import tensorflow_probability as tfp

from misc.autoconfig import ConfiguredModule
from .autoencoder import Model, RegressiveModel, RegressiveAutoencoder
from .utils import *
from misc.latent import NormalSpace, ScaledNormalSpace
'''
The state autoencoder decodes a set of goals embedded into R^lxd
Our data will come in the shape of (batch, goal_max, goal_length_max, dimension)
'''

class StateModel(ConfiguredModule, tf.keras.layers.Layer):
    @property
    def _param_default(self):
        default = dict(
            goal_N = 3, state_N = 3, embed_tokens = 512, goal_querry = 16, state_querry = 8, goal_length = 128, state_length = 128,
            d_model = 128, dff = 512, 
            num_heads = 8, dropout_rate = 0.1, d_scale = 4)
        return default

    def __init__(self, **kwargs):
        '''
        args:
            goal_N: number of goal model layers
            state_N: number of state model layers
            embed_tokens: size of embed dictionary
            goal_querry: number of querry tokens in goal model
            state_querry: number of querry token in state model
            # NOTE the input_shape = (batch_size, goal_length, state_length)
            goal_length: length of each goal # TODO migrate to build()
            state_length: number of goals # TODO migrate
            d_model: number of neurons in NNs and embedding
            dff: see SimpleLayer
            num_heads: number of heads in MHA layers
            dropout_rate: -
            d_scale: # TODO repair
        '''
        super().__init__(param_prefix='_', **kwargs)

        #self.goal_autoencoder = Model('regressive',
        #    N = goal_N, embed_tokens = embed_tokens, querry = goal_querry, output_length = goal_length, 
        #    d_model = d_model, num_heads = num_heads, dff = dff, rate = dropout_rate
        #)
        self.goal_autoencoder = self.configure(Model,'regressive', N=self._goal_N, querry=self._goal_querry)
        #self.state_autoencoder = RegressiveAutoencoder(
        #    N = state_N, querry = state_querry, output_length = state_length, latent_type = 'normal',
        #    d_model = d_model * d_scale, num_heads = num_heads, dff = dff * d_scale, rate = dropout_rate
        #)
        self.state_autoencoder = self.configure(RegressiveAutoencoder, N=self._state_N, querry=self._state_querry)
        self.latent_dense = tf.keras.layers.Dense(self._d_model*self._d_scale)
        self.out_dense = tf.keras.layers.Dense(self._d_model*self._goal_querry)
        #self.goal_latent = NormalSpace(scale_init = 0.15)
        self.goal_latent = ScaledNormalSpace(self._d_model)
        
        self._loss = CategoricalLoss()
        lr = CustomSchedule(self._d_model, 100)
        self.optimizer = tf.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def encode(self, inp, training):
        # inp.shape == (batch_size, state_length, goal_length)
        # TODO state_length and goal_length could be gathered from input and we could avoid explicitly setting them as a parameters
        batch_size = inp.shape[0]
        assert inp.shape[1:] == (self._state_length, self._goal_length)


        state_masks = tf.cast(tf.math.equal(inp[:, :, 1], 0), tf.float32)[:, tf.newaxis, :, tf.newaxis],
        if True:
            look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
            look_ahead_mask = tf.maximum(state_masks[0], look_ahead_mask)
            state_masks = (*state_masks, look_ahead_mask)
        
        x = tf.reshape(inp, (self._state_length*batch_size, self._goal_length))
        goal_outputs, goal_masks = self.goal_autoencoder.encode(x, training)

        goal_latent = goal_outputs[0]
        x = goal_latent.extract(training)
        x = tf.reshape(x, (batch_size, self._state_length, self._goal_querry * self._d_model))
        x = self.latent_dense(x)
        
        state_latent = self.state_autoencoder.encode(x, state_masks[0], training)
        return goal_outputs, (state_latent, x), goal_masks, state_masks
    
    def decode(self, x, mask, training):
        x = self.state_decoder(x, mask, training)
        x = self.out_dense(x)
        x = tf.reshape(x, x.shape[:-1]+(self._goal_querry, self._d_model))
        dist = tfp.distributions.Normal(loc = x, scale = 0.05)
        return dist
        
    
    def __call__(self, inp, target, training):
        assert len(inp.shape)==3
        batch_size = inp.shape[0]

        goal_outputs, state_outputs, goal_masks, state_masks = self.encode(inp, training)
        goal_sample = goal_outputs[0].extract(training)
        state_sample = state_outputs[0].extract(training)

        decoded_from_goals = self.goal_autoencoder.decode(*(goal_sample, *goal_outputs[1:]), *goal_masks, training)
        decoded_from_goals = tf.reshape(decoded_from_goals, (batch_size, self._state_length,) + decoded_from_goals.shape[1:])
        
        goal_latent_reconst_logits = self.state_autoencoder.decode(*(state_sample, *state_outputs[1:]), *state_masks, training)
        goal_latent_reconst_logits = self.out_dense(goal_latent_reconst_logits)
        goal_latent_reconst_logits = tf.reshape(goal_latent_reconst_logits, (batch_size *self._state_length, self._goal_querry, self._d_model))
        goal_latent_reconst = self.goal_latent(goal_latent_reconst_logits)
        goal_latent_reconst_sample = goal_latent_reconst.extract(training)
        goal_latent_reconst_sample = tf.reshape(goal_latent_reconst_sample, (batch_size*self._state_length,) + (self._goal_querry, self._d_model))
        
        decoded_from_latent = self.goal_autoencoder.decode(*(goal_latent_reconst_sample, *goal_outputs[1:]), *goal_masks, training)
        decoded_from_latent = tf.reshape(decoded_from_latent, (batch_size, self._state_length,) + decoded_from_latent.shape[1:])
        
        losses = {}
        losses['goal_autoencoder'] = self._loss(target, decoded_from_goals)
        losses['goal_to_goal'] = self._loss(target, decoded_from_latent)#*((0.25/tf.stop_gradient(losses['goal_embedder']))**1.5)
        losses['state_to_state'] = goal_outputs[0].compare(goal_latent_reconst)
        #self._loss2(goal_embed, goal_embed_reconst)
        scalars = {
                'goal_autoencoder': 2.5,
                'goal_to_goal': 0.5,
                'state_to_state': 0.3,
            }
        loss = 0.0
        for k, l in losses.items():
            loss += scalars[k]*l
        return state_outputs[0], loss, losses
