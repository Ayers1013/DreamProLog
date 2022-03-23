from .autoconfig import ConfiguredModule
from .analytics import AnalyticsModule

import tensorflow as tf

class Module(ConfiguredModule, AnalyticsModule, tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Model(ConfiguredModule, AnalyticsModule, tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)