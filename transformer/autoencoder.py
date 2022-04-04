from misc.autoconfig import ConfiguredModule
from .attention import *
from .utils import *
from misc import latent, Module

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class Encoder(Module):
    @property
    def _param_default(self):
        return dict(collapse_querry=True)

    @property
    def _param_args(self):
        return ['N', 'querry', 'output_length', 'd_model', 'num_heads', 'dff', 'rate']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.positional_encoding = tf.Variable(initial_value = positional_encoding(self._output_length, self._d_model), trainable = False)
        
        shape=(1, self._querry, self._d_model)
        self.pre_variable = tf.Variable(tf.random.uniform(
            shape, minval=0, maxval=1, dtype=tf.dtypes.float32, seed=420, name=None))
        self.q_positional = tf.Variable(positional_encoding(self._querry, self._d_model), trainable = False)
        #self.scale = tf.Variable(initial_value=1., trainable = True)
        self.variable = self.q_positional #* self.scale
        self.variable_sl = MLP(self._d_model, self._dff, self._rate)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [self.configure(SelfAttention) for _ in range(self._N)]
        self.q_layer = self.configure(ConditionedAttention)

        if self._collapse_querry: self.collapse_dense = tf.keras.layers.Dense(self._d_model)

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

        if self._collapse_querry:
            q = tf.concat(q, axis=-2)
            q = self.collapse_dense(q)

        return q, attention

class Decoder(Module):
    @property
    def _param_args(self):
        return ['N', 'output_length', 'd_model', 'num_heads', 'dff', 'rate']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        shape=(1, self._output_length, self._d_model)
        self.variable = tf.Variable(initial_value = positional_encoding(self._output_length, self._d_model), trainable = False)

        self.layers = [self.configure(DeepCrossAttention) for _ in range(self._N)]

    def call(self, x, mask, training):
        batch_size = tf.shape(x)[0]
        v = tf.tile(self.variable, (batch_size, 1, 1))#self.mha(x, x, self.variable)
        q = x

        for l in self.layers:
            v, q, _att = l(v, q, mask, training)

        return v

class RegressiveDecoder(Module):
    @property
    def _param_args(self):
        return ['N', 'output_length', 'd_model', 'num_heads', 'dff', 'rate']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        shape=(1, self._output_length, self._d_model)
        self.variable = tf.Variable(initial_value = positional_encoding(self._output_length, self._d_model), trainable = False)

        #self.mha = MultiHeadAttention(d_model, num_heads)
        self.layers = [self.configure(Attention) for _ in range(self._N)]

    def call(self, x, inp_embed, mask, look_ahead_mask, training):
        v = inp_embed + self.variable
        q = x
        for l in self.layers:
            v, q = l(v, q, mask, look_ahead_mask, training)

        return v

class Autoencoder(Module):
    @property
    def _param_default(self):
        return dict(N=4, querry=8, output_length=128, latent_type='normal', d_model=128, num_heads=4, dff=256, rate=0.04)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.configure(Encoder)
        self.decoder = self.configure(Decoder)

        self.latent_layer = latent.NormalSpace(scale_init = 0.15)

    def encode(self, x, mask, training):
        x, _ = self.encoder(x, mask, training)
        latent = self.latent_layer(x)
        return latent

    def decode(self, x, mask, training):
        x = self.decoder(x, mask, training)
        return x

class RegressiveAutoencoder(Module):
    @property
    def _param_default(self):
        return dict(N=4, querry=8, output_length=128, latent_type = 'normal', d_model=128, num_heads=4, dff=256, rate=0.04, collapse_querry = True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._collapse_querry = True
        self.encoder = self.configure(Encoder) #Encoder(N, querry, output_length, d_model, num_heads, dff, rate, self._collapse_querry)
        self.decoder = self.configure(RegressiveDecoder) #RegressiveDecoder(N, output_length, d_model, num_heads, dff, rate)

        # TODO it was modified by hand
        self.latent_layer = latent.ScaledNormalSpace(self._d_model)#scale_init = 0.15)
        # TODO it was modified once more :(
        #self.latent_layer = latent.NormalSpace(d_model)

        # because we want to use multiple querry during reconstruction
        if self._collapse_querry:
            self.querry_layer = tf.keras.layers.Dense(self._querry*self._d_model)
        
    def encode(self, x, mask, training):
        x, _ = self.encoder(x, mask, training)
        latent = self.latent_layer(x)
        return latent

    def decode(self, x, inp_embed, mask, look_ahead_mask, training):
        if self._collapse_querry:
            x = self.querry_layer(x)
            x = tf.reshape(x, x.shape[:-1]+ (self._querry, self._d_model))
        x = self.decoder(x, inp_embed, mask, look_ahead_mask, training)
        return x

class Model(Module):
    @property
    def _param_args(self):
        return ['model_type']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.autoencoder = self.configure(RegressiveAutoencoder if self._model_type == 'regressive' else Autoencoder)
        self.enc_embed = tf.keras.layers.Embedding(self._embed_tokens, self._d_model)
        self.dense = tf.keras.layers.Dense(self._embed_tokens, activation=None, use_bias=False)

        self.loss = CategoricalLoss()
        self.all_loss = tf.keras.metrics.Sum()
        self.add_loss(lambda: self.all_loss.result())


    def encode(self, inp, training):
        masks = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis],
        inp_embed = self.enc_embed(inp)
        if self._model_type == 'regressive': 
            look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
            dec_target_padding_mask = create_padding_mask(inp)
            look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
            masks = (*masks, look_ahead_mask)

        latent = self.autoencoder.encode(inp_embed, masks[0], training = training)
        outputs = (latent, inp_embed) if self._model_type == 'regressive' else (latent,) 
        return outputs, masks

    def decode(self, *args):
        x = self.autoencoder.decode(*args)
        x = self.dense(x)
        return x

    def call(self, inp, training):
        outputs, masks = self.encode(inp, training)
        #sample latent
        latent = outputs[0]
        x = latent.extract(training)
        y = self.decode(x, *outputs[1:], *masks, training)
        return y
    
    # This method will be the default call method. The only difference is that it returns the latent space instead of the reconstracted input.
    def call_new(self, inp, training):
        outputs, masks = self.encode(inp, training)

        latent = outputs[0]
        x = latent.extract(training)
        y = self.decode(x, *outputs[1:], *masks, training)
        return x, y

    def calc_loss(self, inp, latent, pred):
        self.all_loss.update_state(self.loss(inp, pred))
        
#deprecated
class SetModel(Module):
    @property
    def _param_default(self):
        return dict(N=4, embed_tokens=256, querry=8, output_length=128, latent_type = 'normal', d_model=128, num_heads=4, dff=256, rate=0.04)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.configure(Encoder)
        self.enc_embed = tf.keras.layers.Embedding(self._embed_tokens, self._d_model)
        self.decoder = self.configure(Decoder)

        #TODO add the other latent space options
        self.latent_layer = latent.NormalSpace(scale_init = 0.15)
        
        self.dense = tf.keras.layers.Dense(self._embed_tokens, activation=None, use_bias=False)

    def encode(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        inp_embed = self.enc_embed(inp)
        x, _ = self.encoder(inp_embed, mask, training)
        latent = self.latent_layer(x)
        return latent, mask

    def decode(self, x, mask, training):
        x = self.decoder(x, mask, training)
        x = self.dense(x)
        return x

    def call(self, inp, training):
        latent, mask = self.encode(inp, training)
        x = latent.extract(training)
        x = self.decode(x, mask, training)
        return x

class RegressiveModel(Module):
    @property
    def _param_default(self):
        return dict(N=4, embed_tokens=256, querry=8, output_length=128, latent_type = 'normal', d_model=128, num_heads=4, dff=256, rate=0.04)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.configure(Encoder)  
        self.enc_embed = tf.keras.layers.Embedding(self._embed_tokens, self._d_model)
        self.decoder = self.configure(RegressiveDecoder)

        # TODO add the other latent space options
        self.latent_layer = latent.NormalSpace(scale_init = 0.15)
        
        self.dense = tf.keras.layers.Dense(self._embed_tokens, activation=None, use_bias=False)

    def encode(self, inp, training):
        mask = tf.cast(tf.math.equal(inp, 0), tf.float32)[:, tf.newaxis, :, tf.newaxis]
        look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
        dec_target_padding_mask = create_padding_mask(inp)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        inp_embed = self.enc_embed(inp)
        x, _ = self.encoder(inp_embed, mask, training)
        latent = self.latent_layer(x)
        return latent, inp_embed, mask, look_ahead_mask

    def decode(self, x, inp_embed, mask, look_ahead_mask, training):
        x = self.decoder(x, inp_embed, mask, look_ahead_mask, training)
        x = self.dense(x)
        return x

    def call(self, inp, training):
        latent, inp_embed, mask, look_ahead_mask = self.encode(inp, training)
        x = latent.extract(training)
        x = self.decode(x, inp_embed, mask, look_ahead_mask, training)
        return x