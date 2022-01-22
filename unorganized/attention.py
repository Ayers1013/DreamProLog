import tensorflow as tf
ORDER_RULE_1 = False

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
        if ORDER_RULE_1:
            x = x+self.layernorm2(inp)
        else:
            x = self.layernorm2(inp + x)
        
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
        if ORDER_RULE_1:
            _v = self.sl_v(_v, training)
            if self._add: _v = v + _v
        else:
            if self._add: _v = v + _v
            _v = self.sl_v(_v, training)

        if mask is not None: mask = tf.transpose(mask, (0, 1, 3, 2))
        _q, att_q = self.mha_q(v, v, q, mask)
        if ORDER_RULE_1:
            _q = self.sl_q(_q, training)
            if self._add: _q = q + _q
        else:
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
        if ORDER_RULE_1:
            x = x +  self.sl(_x, training)
        else:
            x = self.sl(_x + x, training)

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
        if ORDER_RULE_1:
            x = x + self.sl1(_x, training)
        else:
            x = self.sl1(_x + x, training)

        _x, _ = self.mha2(y, y, x, mask)
        if ORDER_RULE_1:
            x = x + self.sl2(_x, training)
        else:
            x = self.sl2(_x + x, training)

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
