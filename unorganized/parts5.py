from .transformer import EncoderLayer, DecoderLayer, MultiHeadAttention
import tensorflow as tf
from .misc import positional_encoding
import tensorflow_probability as tfp

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

    self.dropout = tf.keras.layers.Dropout(0.05)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask=None):
    #To support calculations when batxh_size_q=1
    batch_size_q = tf.shape(q)[0]
    batch_size_kv = tf.shape(k)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size_q)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size_kv)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size_kv)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = self.dropout(scaled_attention)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size_kv, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class SimpleLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.04):
        super().__init__()
        self.dense = tf.keras.layers.Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.dense2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, inp, training):
        x = self.dropout(inp, training)
        x = self.layernorm(x)
        
        x = self.dense(x)
        x = self.dense2(x)

        x = self.dropout2(x, training)
        x = self.layernorm2(x+inp) 
        
        return x

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate, add = True):
        super().__init__()
        self._add = add

        self.mha_v = MultiHeadAttention(d_model, num_heads)
        self.mha_q = MultiHeadAttention(d_model, num_heads)

        self.sl_v = SimpleLayer(d_model, dff, rate)
        self.sl_q = SimpleLayer(d_model, dff, rate)

    def call(self, v, q, mask, training):
        'mask:  (batch_size, len_q, len_v)'

        _v, att_v = self.mha_v(q, q, v, mask)
        if self._add: _v = v + _v
        _v = self.sl_v(_v, training)

        if mask is not None: mask = tf.transpose(mask, (0, 1, 3, 2))
        _q, att_q = self.mha_q(v, v, q, mask)
        if self._add: _q = q + _q
        _q = self.sl_q(_q, training)
        return _v, _q, (att_v, att_q)

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.sl = SimpleLayer(d_model, dff, rate)

    def call(self, x, mask, training):

        _x, _att = self.mha(x, x, x, mask)
        x = self.sl(x + _x, training)

        return x, _att
        
class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.sl1 = SimpleLayer(d_model, dff, rate)
        self.sl2 = SimpleLayer(d_model, dff, rate)

    def call(self, x, y, mask, look_ahead_mask, training):
        
        _x, _ = self.mha1(x, x, x, look_ahead_mask)
        x = self.sl1(x + _x, training)

        _x, _ = self.mha2(y, y, x, mask)
        x = self.sl2(x + _x, training)

        return x, y

class DeepCrossAttention(CrossAttention):
    def __init__(self, d_model, num_heads, dff, rate, add = True):
        super().__init__(d_model, num_heads, dff, rate, add)

        #self.satt_v = SelfAttention(d_model, num_heads, dff, rate)
        self.satt_q = SelfAttention(d_model, num_heads, dff, rate)

    def call(self, v, q, mask, training):

        #v = self.satt_v(v, mask, training)

        v, q, att = super().call(v, q, mask, training)
        q, _att = self.satt_q(q, None, training)

        return v, q, att


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
    def __init__(self, N=4, querry=8, output_length=128, d_model=128, num_heads=4, dff=256, rate=0.04):
        super().__init__()
        self.encoder = Encoder(N, querry, output_length, d_model, num_heads, dff, rate)  
        self.enc_embed = tf.keras.layers.Embedding(300, d_model)
        self.decoder = Decoder(N, output_length, d_model, num_heads, dff, rate)

        self.latent_dense = tf.keras.layers.Dense(d_model, activation=None, use_bias=False)
        #self.latent_sl = SimpleLayer(d_model, dff, rate)
        
        self.dense = tf.keras.layers.Dense(300, activation=None, use_bias=False)
        
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
        x, _ = self.encoder(inp_embed, mask, training)
        #_epsilon = 0.02
        #x = x+ _epsilon
        #x/= tf.expand_dims(tf.reduce_sum(x, axis=-1), axis=-1)
        #dist = tfp.distributions.OneHotCategorical(probs = x, dtype=tf.float32)
        #dist = tfp.distributions.RelaxedOneHotCategorical(1., logits = x)
        #x = dist.sample() + dist.probs - tf.stop_gradient(dist.probs)

        x = self.latent_dense(x)

        #x = self.decoder(x, inp_embed, mask, look_ahead_mask, training)
        x = self.decoder(x, mask, training)
        
        x = self.dense(x)
        
        return x




