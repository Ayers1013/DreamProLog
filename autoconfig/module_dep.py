'''
This file contains the main module
'''
import inspect

class ConfigurationSpace:
    def __init__(self, ctor, parent = None, **kwargs):
        signature = inspect.signature(ctor)
        self._parent = parent
        if parent is not None: self._global_config = self._parent._global_config
        self._params = {}
        if self._name in self._global_config:
            self._params.update(self._global_config[self._name])
        for k, v in signature.parameters.items():
            if k not in self._params:
                self._params[k] = self._querry(k, default = v.default)
        self._params.update(kwargs)

    def _querry(self, k, default):
        node = self
        while node is not None:
            if k in node._params:
                return node._params[k]
            node = node._parent
        if default is not inspect._empty:
            return default
        else:
            raise Exception(f'The {k} parameter was not configured for {self._name}.')

class Wrapper(ConfigurationSpace):
    def __init__(self, ctor, parent = None, **kwargs):
        self._name = 'B'
        super().__init__(ctor, parent, **kwargs)
        self._obj = ctor(**self._params)


class Module(ConfigurationSpace):
    def create(self, name, ctor, **kwargs):
        if not hasattr(self, '_modules'):
            self._modules = {}
        self._modules[name] = Wrapper(ctor, self)
        return self._modules[name]

    def get(self, name):
        return self._modules[name]

    def cget(self, name, ctor, **kwargs):
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name in self._modules: return self._modules[name]
        else: return self.create(name, ctor, **kwargs)

class AutoConfigure(Module):
    def __init__(self, name, global_config):
        self._name = name
        self._params = {}
        self._parent = None
        self._global_config = global_config