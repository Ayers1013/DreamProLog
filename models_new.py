import tensorflow as tf

from transformer import StateModel
import networks
import tools

class WorldModel(tools.Module):

  def __init__(self, step, config):
    self._step = step
    self._config = config
    #NOTE to gnn
    self.autoencoder = StateModel()
    self.dense = tf.keras.layers.Dense(128, )
    self.heads = {}

    #NOTE to gnn
    self.heads['reward'] = networks.DenseHead(
      [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
        [], config.discount_layers, config.units, config.act, dist='binary')
    #for name in config.grad_heads:
    #  assert name in self.heads, name

    self._model_opt = tools.Optimizer(
      'model', config.model_lr, config.opt_eps, config.grad_clip,
      config.weight_decay, opt=config.opt)
    
  def train(self, data):
    ep0, act, ep1, meta = data

    with tf.GradientTape() as tape:
      text = ep0['text']
      inp, target = text[:, :, :-1], text[:, :, 1:]
      latent, loss, losses = self.autoencoder(inp, target, True)

      x = latent.extract(True)
      x = tf.reshape(x, (-1, 8*512))
      x = self.dense(x)

      for name, head in self.heads.items():
        grad_head = (name in self._config.grad_heads)
        inp = x # if grad_head else tf.stop_gradient(feat)
        pred = head(inp, tf.float32)
        like = pred.log_prob(tf.cast(ep0[name], tf.float32))
        mse = (tf.cast(ep0[name], tf.float32)-tf.cast(pred.mode(), tf.float32))**2
        losses[name] = tf.reduce_mean(mse)
        loss += mse

    model_parts = [self.autoencoder] + list(self.heads.values())
    varibs = self.trainable_variables
    grads = tape.gradient(loss, varibs)
    #Logginh metrics
    metrics = self._model_opt._opt.apply_gradients(zip(grads, varibs)) # dep: (model_tape, tf.squeeze(loss), model_parts)
    return losses
    #metrics, losses
      

