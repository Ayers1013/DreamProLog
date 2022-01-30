import latent 
import tensorflow as tf

def test_NormalSpace():
    latent_layer = latent.NormalSpace()
    logits = tf.random.unifrom((4, 6))
    x = latent_layer(logits)
    sample = x.sample()
    mode = x.mode()

    print(logits, x, sample, mode, sep='\n\n')

    latent_layer_2 = latent.NormalSpace()
    logits_2 = tf.random.unifrom((4, 6))
    x_2 = latent_layer(logits_2)
    loss = x.comparison_loss(x_2)
    print(loss)

if __name__ == '__main__':
    test_NormalSpace()