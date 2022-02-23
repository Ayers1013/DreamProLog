import tensorflow as tf
from .autoencoder import Model, RegressiveModel
from .models import StateModel

def test_autoencoder_Model(logger):
    shape = (16, 64)
    inp = tf.random.uniform(shape, 1, 100, dtype = tf.int32)
    config = {
        'N' : 3,
        'embed_tokens' : 512,
        'querry' : 16,
        'output_length' : 64,
        'latent_type' : 'normal',
        'd_model' : 128,
        'dff' : 512,
        'num_heads' : 8,
        'rate' : 0.1,
    }

    #model = Model(num_layers, embed_tokens, querry, length, latent, d_model, num_heads, dff, dropout_rate)
    model = Model('set', **config)

    x = model(inp, False)

    assert x.shape == (16, 64, 512)

    return 'transformer.autoencoder.Model checks out.'

def test_autoencoder_RegressiveModel(logger):
    shape = (16, 64)
    inp = tf.random.uniform(shape, 1, 100, dtype = tf.int32)
    config = {
        'N' : 3,
        'embed_tokens' : 512,
        'querry' : 16,
        'output_length' : 64,
        'latent_type' : 'normal',
        'd_model' : 128,
        'dff' : 512,
        'num_heads' : 8,
        'rate' : 0.1,
    }

    #model = RegressiveModel(num_layers, embed_tokens, querry, length, latent, d_model, num_heads, dff, dropout_rate)
    model = Model('regressive', **config)

    x = model(inp, False)

    assert x.shape == (16, 64, 512)

    return 'transformer.autoencoder.RegressiveModel checks out.'

def test_model_StateNet(logger):
    def exp(shape, **kwargs):
        inp = tf.random.uniform(shape, 1, 100, dtype = tf.int32)
        bs, sl, gl = shape
        model = StateModel(state_length=sl, goal_length=gl, **kwargs)

        x = model(inp, inp, False)
        logger(x)
    for shape in [
        (2, 128, 128), (4, 96, 96), (8, 64, 64),
    ]:
      exp(shape)  
    
    config = {'d_model': 32}
    exp(shape, **config)
    config = {'d_model': 32, 'd_scale': 2}
    exp(shape, **config)
    config = {'d_model': 32, 'd_scale': 2, 'state_N': 1, 'goal_querry': 4, 'state_querry': 1, 'dff': 64, 'num_heads': 2}
    exp(shape, **config)

    return 'transformer.model.StateModel checks out.'