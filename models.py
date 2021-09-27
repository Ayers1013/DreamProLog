import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow.python.ops.gen_math_ops import acos_eager_fallback

import networks
import tools


class WorldModel(tools.Module):

  def __init__(self, step, config):
    self._step = step
    self._config = config
    #NOTE to gnn
    self.encoder = networks.Encoder(config=self._config)#networks.DummyEncoder()
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers, config.dyn_shared,
        config.dyn_discrete, config.act, config.dyn_mean_act,
        config.dyn_std_act, config.dyn_min_std, config.dyn_cell)
    self.heads = {}

    #NOTE to gnn
    self.heads['image'] = networks.DummyDecoder()
    self.heads['reward'] = networks.DenseHead(
        [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          [], config.discount_layers, config.units, config.act, dist='binary')
    #for name in config.grad_heads:
    #  assert name in self.heads, name

    self.actor = networks.ActionHead(
        config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)

    self._model_opt = tools.Optimizer(
        'model', config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt)
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)

  def mask_loss(self, inp, target):

    pred=self.actor(inp)
    
    masked_probs=tf.cast(target, dtype=tf.float32)*pred.probs
    probs=tf.reduce_sum(masked_probs, axis=-1)+0.01
    log_probs=tf.math.log(probs)
    loss=tf.reduce_mean(log_probs)
    entropy=tf.reduce_mean(pred.entropy())
    return loss, entropy

  def train(self, data):
    print('Tracing WorldModel train function.')
    data = self.preprocess(data)
    with tf.GradientTape() as model_tape:
      embed, action_embed = self.encoder(data)
      self.actor.feed(action_embed)
      
      #arg_act=tf.math.argmax(data['action'], axis=-1)
      #action=tf.gather(action_embed, data['action'])
      self.dynamics.feed_action_embed(action_embed)
      action=data['action']

      post, prior = self.dynamics.observe(embed, action)
      kl_balance = tools.schedule(self._config.kl_balance, self._step)
      kl_free = tools.schedule(self._config.kl_free, self._step)
      kl_scale = tools.schedule(self._config.kl_scale, self._step)
      kl_loss, kl_value, mse_loss_dyn = self.dynamics.kl_loss(
          post, prior, kl_balance, kl_free, kl_scale)
      feat = self.dynamics.get_feat(post)
      likes = {}
      mse_loss={}
      discount_acc={}
      for name, head in self.heads.items():
        grad_head = (name in self._config.grad_heads)
        inp = feat if grad_head else tf.stop_gradient(feat)
        pred = head(inp, tf.float32)
        like = pred.log_prob(tf.cast(data[name], tf.float32))
        mse=(tf.cast(data[name], tf.float32)-tf.cast(pred.mode(), tf.float32))**2
        mse_loss[name]=tf.reduce_mean(mse)
        if name=='discount':
          sample=pred.sample()
          target=data['discount']
          discount_acc['discount_acc_0']=tf.reduce_sum((1-sample)*(1-target))/tf.reduce_sum(1-target)
          discount_acc['discount_0']=tf.reduce_sum(1-target)
          discount_acc['discount_acc_1']=tf.reduce_sum(sample*target)/tf.reduce_sum(target)
          discount_acc['dicount_1']=tf.reduce_sum(target)
        
        if name in self._config.free_heads:
          like=tf.minimum(like, 4.5)
          '''if name=='image':
            mse=tf.reduce_mean(mse, axis=-1)
          like=tf.where(mse<0.1, tf.ones_like(like)*0.1, like)'''
        likes[name] = tf.reduce_mean(like) * self._scales.get(name, 1.0)
      
      if 'action_mask' in self._config.grad_heads:
        loss, entropy=self.mask_loss(feat, data['axiom_mask'])
        likes['action_mask']=5*loss
        likes['action_mask_entropy']=entropy
      #if likes['image']<-20.:
      #  likes['image']=tf.stop_gradient(likes['image'])
      #NOTE added factor
      head_weigth=0.5
      #model_loss = kl_loss - head_weigth*sum(likes.values())
      model_loss = -likes['discount']
    model_parts = [self.encoder, self.dynamics] + list(self.heads.values())

    #Logginh metrics
    metrics = self._model_opt(model_tape, model_loss, model_parts)
    metrics.update({f'{name}_loss': -like for name, like in likes.items()})
    metrics['mse_dyn']=mse_loss_dyn
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = tf.reduce_mean(kl_value)
    metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy()
    metrics['post_ent'] = self.dynamics.get_dist(post).entropy()
    metrics.update({'mse/'+k : v for k,v in mse_loss.items()})
    metrics.update({k: v for k,v in discount_acc.items()})
    return embed, post, feat, kl_value, metrics, action_embed

  #@tf.function
  def preprocess(self, obs):
    #print('Tracing preprocess')
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    #NOTE We have a simple array
    #obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    #obs['image']=tf.cast(obs['image'],dtype)
    #obs['image']=tf.math.tanh(obs['image']*0.1)
    #obs['reward'] = getattr(tf, self._config.clip_rewards)(obs['reward'])
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
    for key, value in obs.items():
      if key in ['gnn', 'action_space']:
        pass
      else:
        if tf.dtypes.as_dtype(value.dtype) in (
            tf.float16, tf.float32, tf.float64):
          obs[key] = tf.cast(value, dtype)
    return obs

class ImagBehavior(tools.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    self.actor=self._world_model.actor
    self.value = networks.DenseHead(
        [], config.value_layers, config.units, config.act,
        config.value_head)
    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = networks.DenseHead(
          [], config.value_layers, config.units, config.act)
      self._updates = tf.Variable(0, tf.int64)
    kw = dict(wd=config.weight_decay, opt=config.opt)
    self._actor_opt = tools.Optimizer(
        'actor', config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
    self._value_opt = tools.Optimizer(
        'value', config.value_lr, config.opt_eps, config.value_grad_clip, **kw)

  def train(
      self, start, objective=None, imagine=None, tape=None, repeats=None):
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}
    with (tape or tf.GradientTape()) as actor_tape:
      assert bool(objective) != bool(imagine)
      if objective:
        imag_feat, imag_state, imag_action = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats)
        reward = objective(imag_feat, imag_state, imag_action)
      else:
        imag_feat, imag_state, imag_action, reward = imagine(start)
      actor_ent = self.actor(imag_feat, tf.float32).entropy()
      state_ent = self._world_model.dynamics.get_dist(
          imag_state, tf.float32).entropy()
      target, weights = self._compute_target(
          imag_feat, reward, actor_ent, state_ent,
          self._config.slow_actor_target)
      actor_loss, mets = self._compute_actor_loss(
          imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
          weights)
      metrics.update(mets)
    if self._config.slow_value_target != self._config.slow_actor_target:
      target, weights = self._compute_target(
          imag_feat, reward, actor_ent, state_ent,
          self._config.slow_value_target)
    with tf.GradientTape() as value_tape:
      value = self.value(imag_feat, tf.float32)[:-1]
      value_loss = -value.log_prob(tf.stop_gradient(target)) 
      if self._config.value_decay:
        value_loss += self._config.value_decay * value.mode()
      value_loss = tf.reduce_mean(weights[:-1] * value_loss)
    metrics['reward_mean'] = tf.reduce_mean(reward)
    metrics['reward_std'] = tf.math.reduce_std(reward)
    metrics['actor_ent'] = tf.reduce_mean(actor_ent)
    metrics.update(self._actor_opt(actor_tape, actor_loss, [self.actor]))
    metrics.update(self._value_opt(value_tape, value_loss, [self.value]))
    return imag_feat, imag_state, imag_action, weights, metrics

  def _imagine(self, start, policy, horizon, repeats=None):
    dynamics = self._world_model.dynamics
    if repeats:
      start = {k: tf.repeat(v, repeats, axis=1) for k, v in start.items()}
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = tf.stop_gradient(feat) if self._stop_grad_actor else feat
      action = policy(inp).sample()
      #NOTE
      action_embed=dynamics.action_to_embed(action)
      succ = dynamics.img_step(state, action_embed, sample=self._config.imag_sample)
      return succ, feat, action
    feat = 0 * dynamics.get_feat(start)
    
    #NOTE action can be anything 
    action = policy(feat).mode()

    succ, feats, actions = tools.static_scan(
        step, tf.range(horizon), (start, feat, action))
    states = {k: tf.concat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      def unfold(tensor):
        s = tensor.shape
        return tf.reshape(tensor, [s[0], s[1] // repeats, repeats] + s[2:])
      states, feats, actions = tf.nest.map_structure(
          unfold, (states, feats, actions))
    return feats, states, actions

  def _compute_target(self, imag_feat, reward, actor_ent, state_ent, slow):
    reward = tf.cast(reward, tf.float32)
    if 'discount' in self._world_model.heads:
      discount = self._world_model.heads['discount'](
          imag_feat, tf.float32).mean()
    else:
      discount = self._config.discount * tf.ones_like(reward)
    if self._config.future_entropy and tf.greater(
        self._config.actor_entropy(), 0):
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and tf.greater(
        self._config.actor_state_entropy(), 0):
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat, tf.float32).mode()
    else:
      value = self.value(imag_feat, tf.float32).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = tf.stop_gradient(tf.math.cumprod(tf.concat(
        [tf.ones_like(discount[:1]), discount[:-1]], 0), 0))
    return target, weights

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
      weights):
    metrics = {}
    inp = tf.stop_gradient(imag_feat) if self._stop_grad_actor else imag_feat
    policy = self.actor(inp, tf.float32)
    actor_ent = policy.entropy()
    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      imag_action = tf.cast(imag_action, tf.float32)
      actor_target = policy.log_prob(imag_action)[:-1] * tf.stop_gradient(
          target - self.value(imag_feat[:-1], tf.float32).mode())
    elif self._config.imag_gradient == 'both':
      imag_action = tf.cast(imag_action, tf.float32)
      actor_target = policy.log_prob(imag_action)[:-1] * tf.stop_gradient(
          target - self.value(imag_feat[:-1], tf.float32).mode())
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)
    if not self._config.future_entropy and tf.greater(
        self._config.actor_entropy(), 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1]
    if not self._config.future_entropy and tf.greater(
        self._config.actor_state_entropy(), 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]
    actor_loss = -tf.reduce_mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.variables, self._slow_value.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)
