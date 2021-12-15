from abc import ABC

class Space(ABC):
    def __init__(self, **config):
        pass


class AutoConfig:
    def __init__(self, **config):
        self.__configured = False
        for k, v in config.items():
            setattr(self, k ,v)

    def create_config(self):
        pass

class TensorSpace:
    'To handle Tensorflow'
    pass