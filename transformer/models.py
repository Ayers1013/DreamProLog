import tensorflow as tf
import tensorflow_probability as tfp
from .autoencoder import Model, RegressiveModel, RegressiveAutoencoder
from .utils import *
from misc.latent import NormalSpace
'''
The state autoencoder decodes a set of goals embedded into R^lxd
Our data will come in the shape of (batch, goal_max, goal_length_max, dimension)
'''

class StateModel(tf.Module):
    def __init__(self, 
    goal_N = 3, state_N = 3, embed_tokens = 512, goal_querry = 16, state_querry = 8, goal_length = 128, state_length = 128,
    d_model = 128, dff = 512, 
    num_heads = 8, dropout_rate = 0.1, d_scale = 4):
        super().__init__()
        self.goal_querry = goal_querry
        self.goal_autoencoder = Model('regressive',
            N = goal_N, embed_tokens = embed_tokens, querry = goal_querry, output_length = goal_length, 
            d_model = d_model, num_heads = num_heads, dff = dff, rate = dropout_rate
        )
        self.state_autoencoder = RegressiveAutoencoder(
            N = state_N, querry = state_querry, output_length = state_length, latent_type = 'normal',
            d_model = d_model * d_scale, num_heads = num_heads, dff = dff * d_scale, rate = dropout_rate
        )
        self.latent_dense = tf.keras.layers.Dense(d_model*d_scale)
        self.out_dense = tf.keras.layers.Dense(d_model*self.goal_querry)
        self.goal_latent = NormalSpace(scale_init = 0.15)
        
        self._loss = CategoricalLoss()
        lr = CustomSchedule(d_model, 100)
        self.optimizer = tf.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
    def encode(self, inp, training):
        batch_size = inp.shape[0]
        state_masks = tf.cast(tf.math.equal(inp[:, :, 1], 0), tf.float32)[:, tf.newaxis, :, tf.newaxis],
        if True:
            look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
            look_ahead_mask = tf.maximum(state_masks[0], look_ahead_mask)
            state_masks = (*state_masks, look_ahead_mask)
        
        x = tf.reshape(inp, (128*batch_size, 128))
        goal_outputs, goal_masks = self.goal_autoencoder.encode(x, training)

        goal_latent = goal_outputs[0]
        x = goal_latent.extract(training)
        x = tf.reshape(x, (batch_size, 128, self.goal_querry * 128))
        x = self.latent_dense(x)
        
        state_latent = self.state_autoencoder.encode(x, state_masks[0], training)
        return goal_outputs, (state_latent, x), goal_masks, state_masks
    
    def decode(self, x, mask, training):
        x = self.state_decoder(x, mask, training)
        x = self.out_dense(x)
        x = tf.reshape(x, x.shape[:-1]+(self.goal_querry, 128))
        dist = tfp.distributions.Normal(loc = x, scale = 0.05)
        return dist
        
    
    def __call__(self, inp, target, training):
        assert len(inp.shape)==3
        batch_size = inp.shape[0]
        goal_outputs, state_outputs, goal_masks, state_masks = self.encode(inp, training)
        goal_sample = goal_outputs[0].extract(training)
        state_sample = state_outputs[0].extract(training)

        decoded_from_goals = self.goal_autoencoder.decode(*(goal_sample, *goal_outputs[1:]), *goal_masks, training)
        decoded_from_goals = tf.reshape(decoded_from_goals, (batch_size, 128,) + decoded_from_goals.shape[1:])
        
        goal_latent_reconst_logits = self.state_autoencoder.decode(*(state_sample, *state_outputs[1:]), *state_masks, training)
        goal_latent_reconst_logits = self.out_dense(goal_latent_reconst_logits)
        goal_latent_reconst_logits = tf.reshape(goal_latent_reconst_logits, (batch_size *128, 16, 128))
        goal_latent_reconst = self.goal_latent(goal_latent_reconst_logits)
        goal_latent_reconst_sample = goal_latent_reconst.extract(training)
        goal_latent_reconst_sample = tf.reshape(goal_latent_reconst_sample, (batch_size*128,) + (self.goal_querry, 128))
        
        decoded_from_latent = self.goal_autoencoder.decode(*(goal_latent_reconst_sample, *goal_outputs[1:]), *goal_masks, training)
        decoded_from_latent = tf.reshape(decoded_from_latent, (batch_size, 128,) + decoded_from_latent.shape[1:])
        
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
