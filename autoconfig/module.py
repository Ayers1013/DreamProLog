'''
This file contains the main module
'''
import inspect

class ConfigurationSpace:
    def __init__(self, ctor, parent = None, **kwargs):
        signature = inspect.signature(ctor)
        self._parent = parent
        self._params = {}
        for k, v in signature.params:
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

class AutoConfiguration:
    def __init__(self, method, logdir):
        pass

class Wrapper(ConfigurationSpace):
    def __init__(self, ctor, parent = None, **kwargs):
        super().__init__(ctor, parent, **kwargs)
        self._obj = ctor(**self._params)

    def __getattr__(self, name):
        return getattr(self, name)

class Module(ConfigurationSpace):
    def _init_with_autoconfig(self, name, parent):
        signature = inspect.signature(self.__init__)
        super(ConfigurationSpace).__init__(signature, parent)
        self._name = name
        self.__init__(**self._params)

    def create(self, name, ctor, **kwargs):
        if not hasattr(self, '_modules'):
            self._modules = {}
        if isinstance(ctor, Module):
            self._modules[name] = ctor._init_with_autoconfig(self._name + '/' + name, self)
        else:
            self._modules[name] = Wrapper(ctor, self)

    def __getattr__(self, name):
        if name in self._modules: return self._modules[name]
        else: return getattr(self, name)

    def get(self, name):
        return self._modules[name]

    def cget(self, name, ctor, **kwargs):
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name in self._modules: return self._modules[name]
        else:
            self.create(name, ctor, **kwargs)
            return self._modules[name]
           



