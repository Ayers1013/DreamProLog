import tensorflow as tf

class Controller:
  def __init__(self, signature):
    self._signature=signature

  def get_signature(self, batch_size, batch_length):
    _shape=(batch_size, batch_length) if batch_size!=0 else (batch_length,)
    spec=lambda x, dt: tf.TensorSpec(shape=_shape+x, dtype=dt)

    sign=self._signature(batch_size, batch_length)
    sign.update({
      'action': spec((None,), tf.float32),
      'reward': spec((), tf.float32),
      'discount': spec((), tf.float32)
    })

    return sign