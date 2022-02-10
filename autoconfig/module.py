import inspect

class ConfigurationWrapper:
    def __init__(self, ctor, parent = None, name = None, *args, **kwargs):
        signature = inspect.signature(ctor)
        self._parent = parent
        self._root = parent.root
        self._params = {}
        if self._name in self._global_config:
            self._params.update(self._global_config[self._name])
        for k, v in signature.parameters.items():
            if k not in self._params:
                self._params[k] = self._querry(k, default = v.default)
        self._params.update(kwargs)

        self._obj = ctor(*args, **self._params)

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

class ConfiguredModel:
    __slots__ = ['_config']

    def __init__(self):
        'Init should be overwritten.'

    def get(self, name, ctor, *args, **kwargs):
        'Creates or gathers object which is initialized by the autoConfig.'
        self._autoConfig.register(name, ctor, *args, **kwargs)
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules
        return self._modules[name]

class ConfigurationNode:
    def __init__(self, ctor = None, name = None, parent = None, *args, **kwargs):
        'Tree structure:'
        self._parent = parent
        self._core = parent._core if parent is not None else self
        self._children = []

        self._name = name if name is not None else getattr(ctor, '__name__', 'anonym').lower()
        self._params = self._load_params()
        self._params.update(kwargs)

        if ctor is not None:
            signature = inspect.signature(ctor)
            'Load params from args'
            for k, v in zip(signature.parameters, args):
                self._params[k] = v
            for k in signature.parameters:
                if k not in self._params:
                    self._params[k] = self[k]

        'Others:'
        self._cache = None

    def __call__(self,*args, **kwargs):
        'Caches args and kwargs which will be loaded to the next configured object.'
        self._cache = (None, args, kwargs)
        return self

    @property
    def _nested_name(self):
        names = []
        node = self
        while node is not None:
            names.append(node._name)
            node = node._parent
        names.reverse()
        return '/'.join(names)

    def _load_params(self):
        path = []
        node = self
        while node is not None:
            path.append(node._name)
            node = node._parent
        path.reverse()
        config = self._core._get_params_from_core(path)
        return config

    def __sub__(self, ctor):
        'Initializes object with constructor "ctor" using inherited, cached and loaded configurations.'
        name, args, kwargs = self._cache
        self._cache = None
        node = ConfigurationNode(ctor, name, self, *args, **kwargs)
        return ctor(**node._params)


    def __getitem__(self, name):
        'Gather parameters by name.'
        node = self
        while node is not None:
            if name in node._params:
                return node._params[name]
        return None

class ConfigurationCore(ConfigurationNode):
    def __init__(self, name = '', *args, **kwargs):
        self._config_dict = {}
        super().__init__(None, name, None, *args, **kwargs)

    def _get_params_from_core(self, path):
        x = self._config_dict
        for name in path:
            if name not in x: return {}
            x = x[name]
        return {k: v for k, v in x.items() if not isinstance(v, dict)}
