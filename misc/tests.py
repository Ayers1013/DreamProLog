from .latent import *
import tensorflow as tf

def test_latent_NormalSpace(logger):
    latent_layer = NormalSpace()
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()
    mode_2 = x.extract(False)

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = NormalSpace()
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)
    #mode and mode_2 should be equal
    assert tf.reduce_sum(mode-mode_2) <0.01

    return 'misc.latent.NormalSpace checks out.'

def test_latent_ScaledNormalSpace(logger):
    latent_layer = ScaledNormalSpace(6)
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = ScaledNormalSpace(6)
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)

    return 'misc.latent.ScaledNormalSpace checks out.'

def test_latent_DiscrateSpace(logger):
    latent_layer = DiscrateSpace(8, 6)
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = DiscrateSpace(8, 6)
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)

    return 'misc.latent.DiscrateSpace checks out.'

def test_latent_GumbleSpace(logger):
    temp = 0.5
    latent_layer = GumbleSpace(8, 6, temp)
    logits = tf.random.uniform((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    logger(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = GumbleSpace(8, 6, temp)
    logits_2 = tf.random.uniform((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    logger(loss)

    assert sample.shape == (4, 6)
    assert mode.shape == (4, 6)
    assert loss.shape == (4,)

    return 'misc.latent.GumbleSpace checks out.'
