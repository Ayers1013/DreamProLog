import tensorflow as tf

from transformer import StateModel
import networks
import tools
import misc
from transformer.autoencoder import Model

class WorldModel(misc.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.autoencoder = self.configure(Model)
    self._model_opt = tools.Optimizer('model', self._model_lr, self._opt_eps, self._grad_clip, self._weight_decay, opt=self._opt)

  def __call__(self, data, training):
    ep, meta = data
    goals = ep['text']
    inp, target = goals[:, :, :128], goals[:, :, 1:]
    mask = ep['action_mask']

    goal_shape = inp.shape
    inp = tf.reshape(inp, [goal_shape[0]*goal_shape[1]] + goal_shape[2:])
    target = tf.reshape(target, [goal_shape[0]*goal_shape[1]] + goal_shape[2:])

    encoded_goals, pred = self.autoencoder.call_new(inp, training)
    #encoded_goals = tf.reshape(encoded_goals, goal_shape[:2] + (goal_shape[2]*self._d_model,))
    goal_loss = self.autoencoder.loss(target, pred)
    meta_enc, pred = self.autoencoder.call_new(meta[:, :128], training)
    #meta_enc = tf.reshape(meta_enc, (-1, tf.reduce_prod(meta_enc.shape[1:])))
    meta_loss = self.autoencoder.loss(meta[:, 1:], pred)

    mask = ep['action_mask']

    return encoded_goals, meta_enc

    

    

  def train(self, data):
    ep0, act, ep1, meta = data

    losses = {}
    latents = []
    latents_x = []
    with tf.GradientTape() as tape:
      for i, ep in enumerate([ep0, ep1]):
        text = ep['text']
        inp, target = text[:, :, :-1], text[:, :, 1:]
        latent, loss, ep_losses = self.autoencoder(inp, target, True)
        latents.append(latent)
        x = latent.extract(True)
        latents_x.append(x)
        x = tf.reshape(x, (-1, 8*512))
        x = self.dense(x)

        for name, head in self.heads.items():
          grad_head = (name in self._config.grad_heads)
          inp = x # if grad_head else tf.stop_gradient(feat)
          pred = head(inp, tf.float32)
          like = pred.log_prob(tf.cast(ep[name], tf.float32))
          mse = (tf.cast(ep[name], tf.float32)-tf.cast(pred.mode(), tf.float32))**2
          #ep_losses[name] = tf.reduce_mean(mse)
          #loss += -tf.reduce_mean(like) #mse
          ep_losses[name] = -tf.reduce_mean(like)
        losses.update({k+f'_{i}': v for k,v in ep_losses.items()})

      x = latents_x[0] #.extract(True)
      x = tf.reshape(x, (-1, 8*512))
      x = self.dyn(x).sample()
      print(x.shape)
      x = tf.reshape(x, (-1, 8, 512))
      losses['dyn_mse'] = tf.reduce_mean((x-latents_x[1])**2)
      loss = sum(losses.values())

    model_parts = [self.autoencoder] + list(self.heads.values())
    varibs = self.trainable_variables
    grads = tape.gradient(loss, varibs)
    #Logginh metrics
    metrics = self._model_opt._opt.apply_gradients(zip(grads, varibs)) # dep: (model_tape, tf.squeeze(loss), model_parts)
    return losses
    #metrics, losses


