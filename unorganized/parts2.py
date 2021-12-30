from .transformer import EncoderLayer, DecoderLayer, MultiHeadAttention
import tensorflow as tf

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

    def call(self, x, training):
        x = self.dropout(x, training)
        x = self.layernorm(x)
        '''
        x = self.dense(x)
        x = self.dense2(x)

        x = self.dropout2(x, training)
        x = self.layernorm2(x)
        '''
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
        v = self.sl_v(_v, training)

        if mask is not None: mask = tf.transpose(mask, (0, 1, 3, 2))
        _q, att_q = self.mha_q(v, v, q, )
        if self._add: _q = q + _q
        q = self.sl_q(_q, training)
        return v, q, (att_v, att_q)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, querry, d_model, num_heads, dff, rate):
        super().__init__()
        
        shape=(1, querry, d_model)
        self.variable = tf.Variable(tf.random.uniform(
            shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=420, name=None))

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [CrossAttention(d_model, num_heads, dff, rate, add = False) for _ in range(N)]

    def call(self, inp, mask, training):
        batch_size = tf.shape(inp)[0]
        v = inp
        q = tf.tile(self.variable, (batch_size, 1, 1))#self.mha(inp, inp, self.variable)
        #v, q = inp, self.variable

        attention = {}

        for i, l in enumerate(self.layers):
            v, q, att = l(v, q, mask, training)
            attention[i]=att

        return q, attention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, output_length, d_model, num_heads, dff, rate):
        super().__init__()

        shape=(1, output_length, d_model)
        self.variable = tf.Variable(tf.random.uniform(
            shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=420, name=None))

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [CrossAttention(d_model, num_heads, dff, rate, add = True) for _ in range(N)]

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        v = tf.tile(self.variable, (batch_size, 1, 1))#self.mha(x, x, self.variable)
        q = x
        #v, q = self.variable, x

        for l in self.layers:
            v, q, _ = l(v, q, None, training)

        return v

class Net(tf.keras.Model):
    def __init__(self, N=4, querry=8, output_length=128, d_model=128, num_heads=4, dff=256, rate=0.04):
        super().__init__()
        self.encoder = Encoder(N, querry, d_model, num_heads, dff, rate)  
        self.enc_embed = tf.keras.layers.Embedding(300, d_model)
        self.decoder = Decoder(N, output_length, d_model, num_heads, dff, rate)
        
        self.dense = tf.keras.layers.Dense(300, activation=None, use_bias=True)
        
    def call(self, x, training):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        x = self.enc_embed(x)
        
        #mask = tf.expand_dims(mask, axis=-1)
        #mask = tf.tile(mask, (1, 1, 8))
        #mask = tf.expand_dims(mask, axis=1)
        x, _ = self.encoder(x, mask, training)
        x = self.decoder(x, training)
        
        x = self.dense(x)
        
        return x

    def encode(self, x):
        x = self.enc_embed(x)
        x = self.encoder(x, False)

        return x


    def ccall(self, x):
        training = False
        x = self.enc_embed(x)
        x = self.encoder(x, training)
        x = self.decoder(x, training)
        return x



