'''
AnalyticsModule provide functions to log useful metrics in sublayers.
Warning: can only be used if the subclass also inherits from ConfiguredModule.
'''
from matplotlib.pyplot import axis
import tensorflow as tf

ANALYTICS_SWITCH = False

class Analytics:
    'Context manager for turning on and off the AnalyticsModule.'
    def __enter__(self):
        global ANALYTICS_SWITCH
        ANALYTICS_SWITCH = True

    def __exit__(self, *args):
        global ANALYTICS_SWITCH
        ANALYTICS_SWITCH = False


class AnalyticsModule:
    def __init__(self) -> None:
        super().__init__()
        self.__analytics = {}

    def log_analytics(self, name, value):
        if ANALYTICS_SWITCH:
            self.__analytics[name] = value

    def update_analytics(self, upd):
        if ANALYTICS_SWITCH:
            self.__analytics.update(upd)

    def log_vector_analytics(self, tensor, name):
        if ANALYTICS_SWITCH:
            metrics = analyse_vector(tensor, name)
            self.update_analytics(metrics)

    def log_matrix_analytics(self, tensor, name):
        if ANALYTICS_SWITCH:
            metrics = analyse_matrix(tensor, name)
            self.update_analytics(metrics)

    @property
    def analytics(self):
        collected_logs = {}
        for node in self:
            if isinstance(node, AnalyticsModule):
                name = node._nested_name
                collected_logs.update({name + '/' + k: v for k,v in node.__analytics.items()})

        return collected_logs

def analyse_vector(tensor, name=''):
    metrics = {}

    metrics[name + '_norm'] = tf.reduce_mean(tf.norm(tensor, axis=-1))
    metrics[name + '_mean_norm'] = tf.norm(tf.reduce_mean(tensor, axis=-1))
    metrics[name + '_std_norm'] = tf.norm(tf.math.reduce_std(tensor, axis=-1))

    return metrics

def analyse_matrix(tensor, name=''):
    metrics = {}

    metrics[name + '_rank'] = tf.reduce_mean(tf.linalg.matrix_rank(tensor))

    return metrics

def analyse_matrix_dep(tensor, name=''):
    metrics = {}

    metrics[name + '_norm'] = tf.reduce_mean(tf.norm(tensor, axis=(-2, -1)))

    size = tf.cast(tf.reduce_max(tensor.shape[-2:]), tf.float32)
    s = tf.linalg.svd(tensor, compute_uv=False)
    max_s = tf.reshape(tensor, [-1] + [i for i in range(len(tensor.shape)-3)])[0]
    mean_s = tf.reduce_mean(max_s)
    metrics[name + '_mean_max_singular_value'] = mean_s
    metrics[name + '_std_max_singular_value'] = tf.math.reduce_std(max_s)
    for i, eps in enumerate([4e3, 1e3, 4e4, 1e4]):
        metrics[f'name_rank_{i}'] = tf.reduce_mean(tf.linalg.matrix_rank(tensor, tol=size*mean_s*eps))

    return metrics



    
