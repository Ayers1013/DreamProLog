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

class EncoderStartLayer(tf.keras.layers.Layer):
    def __init__(self, querry, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        shape=(1, querry, d_model)
        self.variable = tf.Variable(tf.random.uniform(
            shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=69, name=None))

    def call(self, inp, training):
        x, _ = self.mha(inp, inp, self.variable) #(batch_size, querry, d_model)
        x = self.layernorm1(x)
        x = self.dropout1(x, training=training)

        x = self.ffn(x)
        x = self.layernorm2(x)
        x = self.dropout2(x, training=training)

        return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, inp, training, mask=None):

    attn_output1, _ = self.mha1(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output1 = self.dropout1(attn_output1, training=training)

    attn_output2, _ = self.mha2(inp, inp, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output2 = self.dropout1(attn_output2, training=training)
    out1 = self.layernorm1(x + attn_output1 + attn_output2)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class DecoderStartLayer(tf.keras.layers.Layer):
    def __init__(self, output_length, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        shape=(1, output_length, d_model)
        self.variable = tf.Variable(tf.random.uniform(
            shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=69, name=None))


    def call(self, enc_output, training):
        x, _ = self.mha(enc_output, enc_output, self.variable)
        x = self.layernorm1(x)
        x = self.dropout1(x, training=training)

        x = self.ffn(x)
        x = self.layernorm2(x)
        x = self.dropout2(x, training=training)
        return x
        

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask=None, padding_mask=None):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3


class Encoder(tf.keras.layers.Layer):
    def __init__(self, querry, d_model, num_heads, dff, rate):
        super().__init__()
        self.start_layer = EncoderStartLayer(querry, d_model, num_heads, dff, rate)
        self.layers = [EncoderLayer(d_model, num_heads, dff, rate) for i in range(4)]
        
        
    def call(self, inp, training):
        x = self.start_layer(inp, training)
        for layer in self.layers:
            x = layer(x, inp, training, None)
        
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_length, d_model, num_heads, dff, rate):
        super().__init__()
        self.start_layer = DecoderStartLayer(output_length, d_model, num_heads, dff, rate)
        self.layers = [DecoderLayer(d_model, num_heads, dff, rate) for i in range(4)]
        
        
    def call(self, x, training):
        y = self.start_layer(x, training)
        for layer in self.layers:
            y = layer(y, x, training, None, None)
            
        return y
        
class Net(tf.keras.Model):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.encoder = Encoder(8, d_model, num_heads, dff, 0.04)  
        self.enc_embed = tf.keras.layers.Embedding(300, d_model)
        self.decoder = Decoder(128, d_model, num_heads, dff, 0.04)
        
        self.dense = tf.keras.layers.Dense(300, activation='relu', use_bias=False)
        
    def call(self, x, training):
        training = False
        x = self.enc_embed(x)
        x = self.encoder(x, training)
        x = self.decoder(x, training)
        
        x = self.dense(x)
        x = x*2.
        
        return x