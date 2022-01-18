from .attention import *
form .misc import positional_encoding

import tensorflow as tf
import tensorflow_probability as tfp

class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, querry, output_length, d_model, num_heads, dff, rate):
        super().__init__()

        self.positional_encoding = tf.Variable(initial_value = positional_encoding(output_length, d_model), trainable = False)
        
        shape=(1, querry, d_model)
        self.pre_variable = tf.Variable(tf.random.uniform(
            shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=420, name=None))
        self.q_positional = tf.Variable(positional_encoding(querry, d_model), trainable = False)
        #self.scale = tf.Variable(initial_value=1., trainable = True)
        self.variable = self.q_positional #* self.scale
        self.variable_sl = SimpleLayer(d_model, dff, rate)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [SelfAttention(d_model, num_heads, dff, rate) for _ in range(N)]
        self.q_layer = DeepCrossAttention(d_model, num_heads, dff, rate, add = True)

        latent_tokens = 256
        self.dense = tf.keras.layers.Dense(latent_tokens, use_bias = False)

    def call(self, inp, mask, training):
        batch_size = tf.shape(inp)[0]
        v = inp + self.positional_encoding

        variable = self.variable_sl(self.variable + self.pre_variable)
        q = tf.tile(variable, (batch_size, 1, 1))#self.mha(inp, inp, self.variable)
        #v, q = inp, self.variable

        attention = {}

        for i, l in enumerate(self.layers):
            v, att = l(v, mask, training)
            attention[i]=att

        v, q, _ = self.q_layer(v, q, mask, training)
        #_mask = 1 - tf.squeeze(mask, axis=1)
        #q = tf.reduce_sum(v*_mask, axis=-2)/tf.expand_dims(tf.reduce_sum(_mask, axis=1), axis=1)

        #x = self.dense(q)
        #x = tf.math.softmax(x, axis=-1)
        x = q
        return x, attention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, output_length, d_model, num_heads, dff, rate):
        super().__init__()

        shape=(1, output_length, d_model)
        self.variable = tf.Variable(initial_value = positional_encoding(output_length, d_model), trainable = False)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [DeepCrossAttention(d_model, num_heads, dff, rate, add = True) for _ in range(N)]

    def call(self, x, mask, training):
        batch_size = tf.shape(x)[0]
        v = tf.tile(self.variable, (batch_size, 1, 1))#self.mha(x, x, self.variable)
        q = x
        #v, q = self.variable, x

        for l in self.layers:
            v, q, _ = l(v, q, mask, training)

        return v

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class RegressiveDecoder(tf.keras.layers.Layer):
    def __init__(self, N, output_length, d_model, num_heads, dff, rate):
        super().__init__()

        shape=(1, output_length, d_model)
        self.variable = tf.Variable(initial_value = positional_encoding(output_length, d_model), trainable = False)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [Attention(d_model, num_heads, dff, rate) for _ in range(N)]

    def call(self, x, inp, mask, look_ahead_mask, training):

        v = inp + self.variable
        #q = x
        q = v
        #mask = tf.transpose(mask, (0, 1, 3, 2))

        for l in self.layers:
            v, q = l(v, q, mask, look_ahead_mask, training)

        return v

class Net(tf.keras.Model):
    def __init__(self, N=4, embed_tokens=256, querry=8, output_length=128, d_model=128, num_heads=4, dff=256, rate=0.04):
        super().__init__()
        self.encoder = Encoder(N, querry, output_length, d_model, num_heads, dff, rate)  
        self.enc_embed = tf.keras.layers.Embedding(embed_tokens, d_model)
        self.decoder = Decoder(N, output_length, d_model, num_heads, dff, rate)

        self.latent_dense = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        #self.latent_sl = SimpleLayer(d_model, dff, rate)
        
        self.dense = tf.keras.layers.Dense(embed_tokens, activation=None, use_bias=False)

    def encode(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        inp_embed = self.enc_embed(inp)
        x, _ = self.encoder(inp_embed, mask, training)
        return x, mask

    def decode(self, x, mask, training):
        x = self.latent_dense(x)
        x = self.decoder(x, mask, training)
        x = self.dense(x)
        return x

        
    def call(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        inp_embed = self.enc_embed(inp)

         
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        #look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
        #dec_target_padding_mask = create_padding_mask(inp)
        #look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        #mask = tf.expand_dims(mask, axis=-1)
        #mask = tf.tile(mask, (1, 1, 8))
        #mask = tf.expand_dims(mask, axis=1)
        latent, _ = self.encoder(inp_embed, mask, training)
        #_epsilon = 0.02
        #x = x+ _epsilon
        #x/= tf.expand_dims(tf.reduce_sum(x, axis=-1), axis=-1)
        #dist = tfp.distributions.OneHotCategorical(probs = x, dtype=tf.float32)
        #dist = tfp.distributions.RelaxedOneHotCategorical(1., logits = x)
        #x = dist.sample() + dist.probs - tf.stop_gradient(dist.probs)

        x = self.latent_dense(latent)

        #x = self.decoder(x, inp_embed, mask, look_ahead_mask, training)
        x = self.decoder(x, mask, training)
        x = self.dense(x)
        
        return x