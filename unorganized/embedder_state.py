from .attention import *
import tensorflow as tf
import tensorflow_probability as tfp
from .embedder_goal import Net, Decoder, Encoder 
from .misc import Loss, CustomSchedule

'''
The state autoencoder decodes a set of goals embedded into R^lxd
Our data will come in the shape of (batch, goal_max, goal_length_max, dimension)
'''

class StateNet(tf.keras.Model):
    def __init__(self, ):
        super().__init__()
        num_layers = 3
        embed_tokens = 512
        querry = 16
        self.goal_querry = querry
        length = 128
        d_model = 128
        dff = 512
        num_heads = 8
        dropout_rate = 0.1
        d_scale = 4
        self.goal_embedder = Net(num_layers, embed_tokens, querry, length, d_model, num_heads, dff, dropout_rate)
        self.latent_dense = tf.keras.layers.Dense(d_model*d_scale)
        self.state_encoder = Encoder(N = 3, querry = 16, output_length = 128, 
                                     d_model = d_model*d_scale, num_heads = num_heads, dff = dff*d_scale, rate = dropout_rate)
        self.state_decoder = Decoder(N = 3, output_length = 128, 
                                     d_model = d_model*d_scale, num_heads = num_heads, dff = dff*d_scale, rate = dropout_rate)
        self.out_dense = tf.keras.layers.Dense(d_model*self.goal_querry)
        
        self._loss = Loss()
        self._loss2 = tf.keras.losses.MeanSquaredError()
        lr = CustomSchedule(d_model, 1000)
        self._opt = tf.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
    def encode(self, inp, mask, training):
        batch_size = inp.shape[0]
        x = tf.reshape(inp, (128*batch_size, 128))
        goal_embed, goal_mask = self.goal_embedder.encode(x, training)
        #goal_embed = tf.reshape(x, (batch_size, 128) + (8, 128))
        x = tf.reshape(goal_embed, (batch_size, 128, self.goal_querry * 128))
        x = self.latent_dense(x)
        
        y, _ = self.state_encoder(x, mask, training)
        return y, goal_embed, goal_mask
    
    def decode(self, x, mask, training):
        x = self.state_decoder(x, mask, training)
        x = self.out_dense(x)
        x = tf.reshape(x, x.shape[:-1]+(self.goal_querry, 128))
        return x
        
    
    def calc_loss(self, inp, target, training):
        mask = tf.cast(tf.math.equal(inp[:, :, 1], 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        latent, goal_embed, goal_mask = self.encode(inp, mask, training)
        decoded_from_goals = self.goal_embedder.decode(goal_embed, goal_mask, training)
        decoded_from_goals = tf.reshape(decoded_from_goals, (4, 128,) + decoded_from_goals.shape[1:])
        goal_embed_reconst = self.decode(latent, mask, training)
        goal_embed_reconst = tf.reshape(goal_embed_reconst, (4*128,) + goal_embed_reconst.shape[2:])
        decoded_from_embed = self.goal_embedder.decode(goal_embed_reconst, goal_mask, training)
        decoded_from_embed = tf.reshape(decoded_from_embed, (4, 128,) + decoded_from_embed.shape[1:])
        
        losses = {}
        losses['goal_embedder'] = self._loss(target, decoded_from_goals)
        losses['goal_to_goal'] = self._loss(target, decoded_from_embed)*((0.25/tf.stop_gradient(losses['goal_embedder']))**1.5)
        losses['state_to_state'] = self._loss2(goal_embed, goal_embed_reconst)
        return losses
    
    def call(self, x, training):
        inp, target = x
        losses = self.calc_loss(inp, target, training)
        scalars = {
                'goal_embedder': 2.5,
                'goal_to_goal': 1.,
                'state_to_state': 0.3,
            }
        loss = 0.0
        for k, l in losses.items():
            loss += scalars[k]*l
        return loss
    
    #@tf.function(input_signature=(sgn, sgn))
    def train(self, inp, target):
        with tf.GradientTape() as tape:
            losses = self.loss(inp, target, True)
            scalars = {
                'goal_embedder': 2.5,
                'goal_to_goal': 0.2,
                'state_to_state': 0.5,
            }
            loss = 0.0
            for k, l in losses.items():
                loss += scalars[k]*l
               
        return self._opt(tape, 4*loss, [self])
               
    

