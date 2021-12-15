from .space import Space, AutoConfig

import tensorflow as tf
from typing import List

class SpaceEucledian(Space, AutoConfig):
    def __init__(self, dim: int):
        self.dim = dim

class MorphismEucledian(AutoConfig):
    def __init__(self, object, **config):
        super(AutoConfig).__init__()

    def _create(self):
        assert self.__configured

        self.dense_nets = []
        pass

class UniformDense:
    def __init__(self, layers: int, **layer_config):
        self._dense_nets = [tf.keras.layers.Dense(**layer_config)] 

    @staticmethod
    def get_config_skeleton():
        return {
            'units': int,
            
            }

class DefaultDense:
    def __init__(self, layers: int):
        pass



