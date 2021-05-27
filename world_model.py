import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

import networks
import tools
import networks_ProLog


class WorldModel(tools.Module):
  def __init__(self, step, config):
    self._step = step
    self._config = config
    #NOTE to gnn
    self.encoder = networks_ProLog.Encoder(input_pipes=config.input_pipes, action_embed=config.action_embed)#networks_ProLog.DummyEncoder()

    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers, config.dyn_shared,
        config.dyn_discrete, config.act, config.dyn_mean_act,
        config.dyn_std_act, config.dyn_min_std, config.dyn_cell)
    self.heads = {}
    channels = (1 if config.atari_grayscale else 3)
    shape = config.size + (channels,)

    #NOTE to gnn
    self.heads['image'] = networks_ProLog.DummyDecoder()
    self.heads['reward'] = networks.DenseHead(
        [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          [], config.discount_layers, config.units, config.act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    self._model_opt = tools.Optimizer(
        'model', config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt)
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)

  def train(self, data):
    data = self.preprocess(data)
    with tf.GradientTape() as model_tape:
      embed, action_embed = self.encoder(data)
      post, prior = self.dynamics.observe(embed, data['action'])
      kl_balance = tools.schedule(self._config.kl_balance, self._step)
      kl_free = tools.schedule(self._config.kl_free, self._step)
      kl_scale = tools.schedule(self._config.kl_scale, self._step)
      kl_loss, kl_value = self.dynamics.kl_loss(
          post, prior, kl_balance, kl_free, kl_scale)
      feat = self.dynamics.get_feat(post)
      likes = {}
      for name, head in self.heads.items():
        grad_head = (name in self._config.grad_heads)
        inp = feat if grad_head else tf.stop_gradient(feat)
        pred = head(inp, tf.float32)
        like = pred.log_prob(tf.cast(data[name], tf.float32))
        likes[name] = tf.reduce_mean(like) * self._scales.get(name, 1.0)
      #NOTE Temporary no kl_loss (inactivated)
      model_loss = kl_loss - sum(likes.values())
    model_parts = [self.encoder, self.dynamics] + list(self.heads.values())
    metrics = self._model_opt(model_tape, model_loss, model_parts)
    metrics.update({f'{name}_loss': -like for name, like in likes.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = tf.reduce_mean(kl_value)
    metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy()
    metrics['post_ent'] = self.dynamics.get_dist(post).entropy()
    return embed, post, feat, kl_value, metrics

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    #NOTE We have a simple array
    #obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    obs['image']=tf.cast(obs['image'],dtype)
    obs['image']=tf.math.tanh(obs['image']*0.1)
    obs['reward'] = getattr(tf, self._config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
    for key, value in obs.items():
      if tf.dtypes.as_dtype(value.dtype) in (
          tf.float16, tf.float32, tf.float64):
        obs[key] = tf.cast(value, dtype)
    return obs
