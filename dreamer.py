import collections

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec


from tensorflow_probability import distributions as tfd



import exploration as expl
import models
import tools

from envs import ProLog
pl=ProLog.ProLog()
_signature=pl.output_sign
def get_signature(batch_size, batch_length):
    _shape=(batch_size, batch_length) if batch_size!=0 else (batch_length,)
    spec=lambda x, dt: tf.TensorSpec(shape=_shape+x, dtype=dt)

    sign=_signature(batch_size, batch_length)
    sign.update({
      'action': spec((), tf.int32),
      'reward': spec((), tf.float32),
      'discount': spec((), tf.float32)
    })

    return sign

sign=get_signature(8,2)


class Dreamer(tools.Module):

  def __init__(self, config, logger, dataset):
    self._config = config
    self._logger = logger
    self._float = prec.global_policy().compute_dtype
    #NOTE We cant log the video
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(config.traindir), dtype=tf.int64)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = iter(dataset)
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.    
    x=next(self._dataset)
    #NOTE it is better to initials everything in advance
    #self._train(x)

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step.numpy().item()
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      #NOTE feat, action=state and action is integer which cause problems
      state = tf.nest.map_structure(lambda x: x * tf.cast(mask, dtype=x.dtype), state[0]), tf.zeros_like(state[1])

    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, mean in self._metrics.items():
          self._logger.scalar(name, float(mean.result()))
          mean.reset_states()
        #openl = self._wm.video_pred(next(self._dataset))
        #self._logger.video('train_openl', openl)
        self._logger.write(fps=False)
    action, state = self._policy(obs, state, training)
    if training:
      self._step.assign_add(len(reset))
      self._logger.step = self._step.numpy().item()
    return action, state

  #@tf.function(experimental_relax_shapes=True)
  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      #print('The state was NONE')
      action = tf.zeros(shape=(1), dtype=tf.int32)#tf.zeros(shape=(1, 64), dtype=tf.float32) #tf.zeros((batch_size, obs['action_space'].shape[1]), self._float)
    else:
      latent, action = state
    embed, action_embed = self._wm.encoder(self._wm.preprocess(obs))
    #feed the actor head with the embedding
    self._task_behavior.actor.feed(action_embed)
    self._wm.dynamics.feed_action_embed(action_embed)
    embeded_action=self._wm.dynamics.action_to_embed(action)
    #embeded_action=tf.squeeze(embeded_action, axis=0)
    #print('NOTE: action.shape; ', action.shape, ';', action_embed.shape)
    #print('Observation meta:', obs['axiom_mask'].shape, obs['action_space'][0]['num_clauses'], obs['action_space'][0]['num_nodes'])
    #embeded_action=tf.reshape(tf.squeeze(embeded_action), [-1, 64])

    latent, _ = self._wm.dynamics.obs_step(
        latent, embeded_action, embed, self._config.collect_dyn_sample)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)#, action_embed
    if not training:
      action = self._task_behavior.actor(feat).mode()
    elif self._should_expl(self._step):
      action = self._expl_behavior.actor(feat).sample()
    else:
      action = self._task_behavior.actor(feat).sample()
    if self._config.actor_dist == 'onehot_gumble':
      action = tf.cast(
          tf.one_hot(tf.argmax(action, axis=-1), self._config.num_actions),
          action.dtype)
    action = self._exploration(action, training)
    state = (latent, action)
    return action, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    amount = tf.cast(amount, self._float)
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  @tf.function(input_signature=[sign])
  def _train(self, data):
    print('Tracing train function.')
    metrics = {}
    embed, post, feat, kl, mets, action_embed = self._wm.train(data)
    #feed the actor head with the embedding. It has been moved to WM
    #self._task_behavior.actor.feed(action_embed)

    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      embed, feat, kl = embed[:, :-1], feat[:, :-1], kl[:, :-1]
    reward = lambda f, s, a: self._wm.heads['reward'](f).mode()
    metrics.update(self._task_behavior.train(start, reward)[-1])
    if self._config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, feat, embed, kl)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    for name, value in metrics.items():
      self._metrics[name].update_state(value)

  @tf.function(input_signature=[sign])
  def _train_only_wordModel(self, data):
    print('Tracing train only WordModel function.')
    metrics = {}
    embed, post, feat, kl, mets, action_embed = self._wm.train(data)
    metrics.update(mets)

    for name, value in metrics.items():
      self._metrics[name].update_state(value)



def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))