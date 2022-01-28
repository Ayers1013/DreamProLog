from .attention import *
from .util import *
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

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
        self.variable_sl = MLP(d_model, dff, rate)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [SelfAttention(d_model, num_heads, dff, rate) for _ in range(N)]
        self.q_layer = ConditionedAttention(d_model, num_heads, dff, rate)

        latent_tokens = 256
        self.dense = tf.keras.layers.Dense(latent_tokens, use_bias = False)

    def call(self, inp, mask, training):
        batch_size = tf.shape(inp)[0]
        v = inp + self.positional_encoding

        variable = self.variable_sl(self.variable + self.pre_variable)
        q = tf.tile(variable, (batch_size, 1, 1))

        attention = {}

        for i, l in enumerate(self.layers):
            v, att = l(v, mask, training)
            attention[f'layer_att_{i}']=att

        q, _att = self.q_layer(v, q, mask, training)
        attention['q_layer_att'] = _att
        return q, attention

class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, output_length, d_model, num_heads, dff, rate):
        super().__init__()

        shape=(1, output_length, d_model)
        self.variable = tf.Variable(initial_value = positional_encoding(output_length, d_model), trainable = False)

        self.layers = [DeepCrossAttention(d_model, num_heads, dff, rate) for _ in range(N)]

    def call(self, x, mask, training):
        batch_size = tf.shape(x)[0]
        v = tf.tile(self.variable, (batch_size, 1, 1))#self.mha(x, x, self.variable)
        q = x

        for l in self.layers:
            v, q, _att = l(v, q, mask, training)

        return v

class RegressiveDecoder(tf.keras.layers.Layer):
    def __init__(self, N, output_length, d_model, num_heads, dff, rate):
        super().__init__()

        shape=(1, output_length, d_model)
        self.variable = tf.Variable(initial_value = positional_encoding(output_length, d_model), trainable = False)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [Attention(d_model, num_heads, dff, rate) for _ in range(N)]

    def call(self, x, inp, mask, look_ahead_mask, training):

        v = inp + self.variable
        q = x
        #q = v
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

        self.latent_dense1 = tf.keras.layers.Dense(d_model, activation = None)
        self.latent_dense2 = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        self.scale_dense = tf.keras.layers.Dense(d_model, activation = 'sigmoid')
        #self.latent_sl = SimpleLayer(d_model, dff, rate)
        
        self.dense = tf.keras.layers.Dense(embed_tokens, activation=None, use_bias=False)

    def encode(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        inp_embed = self.enc_embed(inp)
        x, _ = self.encoder(inp_embed, mask, training)
        #dist = tfd.Normal(loc = x, scale = 0.05)
        return x, mask

    def decode(self, x, mask, training):
        x = self.latent_dense2(x)
        x = self.decoder(x, mask, training)
        x = self.dense(x)
        return x

    def call(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        inp_embed = self.enc_embed(inp)
        latent, _ = self.encoder(inp_embed, mask, training)
        dist = tfd.Normal(loc = latent, scale = 0.05)
        if training:
            latent = dist.sample()
        else:
            latent = dist.mode()

        x = self.latent_dense2(latent)
        x = self.decoder(x, mask, training)
        x = self.dense(x)
        
        return x

class NetII(tf.keras.Model):
    def __init__(self, N=4, embed_tokens=256, querry=8, output_length=128, d_model=128, num_heads=4, dff=256, rate=0.04):
        super().__init__()
        self.encoder = Encoder(N, querry, output_length, d_model, num_heads, dff, rate)  
        self.enc_embed = tf.keras.layers.Embedding(embed_tokens, d_model)
        self.decoder = RegressiveDecoder(N, output_length, d_model, num_heads, dff, rate)

        self.latent_dense1 = tf.keras.layers.Dense(d_model, activation = None)
        self.latent_dense2 = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        self.scale_dense = tf.keras.layers.Dense(d_model, activation = 'sigmoid')
        #self.latent_sl = SimpleLayer(d_model, dff, rate)
        
        self.dense = tf.keras.layers.Dense(embed_tokens, activation=None, use_bias=False)

    def encode(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        inp_embed = self.enc_embed(inp)
        x, _ = self.encoder(inp_embed, mask, training)
        #dist = tfd.Normal(loc = x, scale = 0.05)
        return x, mask

    def decode(self, x, mask, training):
        x = self.latent_dense2(x)
        x = self.decoder(x, mask, training)
        x = self.dense(x)
        return x

    def call(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
        dec_target_padding_mask = create_padding_mask(inp)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        inp_embed = self.enc_embed(inp)
        latent, _ = self.encoder(inp_embed, mask, training)
        dist = tfd.Normal(loc = latent, scale = 0.05)
        if training:
            latent = dist.sample()
        else:
            latent = dist.mode()

        x = self.latent_dense2(latent)
        x = self.decoder(x, inp_embed, mask, look_ahead_mask, training)
        x = self.dense(x)
        
        return x