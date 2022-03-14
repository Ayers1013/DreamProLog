import inspect
import pathlib

import ruamel.yaml as yaml

GET_PARAMETER_BY_ATTR = True
DEEP_NAME_STRUCTURE = True

def update_nested(to_dict, from_dict):
    for k, v in from_dict.items():
        if isinstance(v, dict):
            if k not in to_dict: to_dict[k] = v
            else:
                to_dict[k] = {}
                update_nested(to_dict[k], v)
        else:
            to_dict[k] = v

class Configuration:
    def __init__(self, config, **kwargs):
        self.__params = {}
        self.load(config)
        self.update(kwargs)

    def load(self, config):
        if isinstace(config, str):
            _, ext = config.split('.')
            if ext == 'yaml':
                self.load_from_yaml(config)
        
        elif isinstance(config, list):
            for c in config:
                self.load(c)


    def load_from_yaml(path):
        self.__params.update(yaml.safe_load((pathlib.Path(path)).read_text()))


class ConfigurationNode:
    def __init__(self, parent=None, config_name=None, unique_name=None, params=None):
        'Tree structure:'
        super().__init__()
        self.__parent = parent
        self.__core = parent.__core if parent is not None else self
        self.__children = {}
        self.__config_name = config_name
        self.__unique_name = unique_name

        if parent is not None: parent.__children[self.__unique_name] = self

        self.__params = self.__load_params()
        #print(config_name, '----\n', params, self.__params, '\n----\n')
        if params is not None: self.__params.update(params)

    @property
    def __nested_name(self):
        names = []
        node = self
        while node is not None:
            names.append(node.__name)
            node = node.__parent
        names.reverse()
        return '/'.join(names)

    def __load_params(self):
        if self.__parent is None: return {}
        if self.__unique_name in self.__parent.__params:
            return self.__parent.__params[self.__unique_name]
        elif self.__config_name in self.__parent.__params:
            return self.__parent.__params[self.__config_name]        
        return {}

    '''
    def __load_params(self):
        if self.__parent is None: return {}
        params = {}
        if self.__config_name in self.__parent.__params:
            params = self.__parent.__params[self.__config_name] 
        if self.__unique_name in self.__parent.__params:
            update_nested(params, self.__parent.__params[self.__unique_name])
        return params
    '''

    def __getattr__(self, name):
        if GET_PARAMETER_BY_ATTR and name[0] != '_':
            param = self[name]
            if param is not None: return param
        raise AttributeError

    def __getitem__(self, name):
        'Gather parameters by name.'
        node = self
        while node is not None:
            if name in node.__params:
                return node.__params[name]
            node = node.__parent
        return None

    def configure(self, ctor, *args, config_name=None, unique_name=None, **kwargs):
        config_name=config_name if config_name is not None else ctor.__name__
        if unique_name is None:
            index = [child.__config_name for child in self.__children.values()].count(config_name)
            unique_name = config_name + f'_{index}'

        if issubclass(ctor, ConfiguredModule):
            return ctor(*args, parent=self, config_name=config_name, unique_name=unique_name, **kwargs)
        else:
            if DEEP_NAME_STRUCTURE:
                node = ConfigurationNode(self, config_name, unique_name)
                node.__params.update(kwargs)

            ckwargs = {}
            signature = inspect.signature(ctor)
            for k in list(signature.parameters)[len(args):]:
                if k == 'args' or k == 'kwargs': continue
                param = self[k]
                if param is not None:
                    ckwargs[k] = params
            return ctor(*args, **ckwargs)

    def _stats(self):
        return (self.__config_name, self.__unique_name, self.__params, self.__parent)

    def _name_structure(self, suff = '', include_config_name = True, display_params = True, full_depth = True):
        res = suff
        if include_config_name: res+= f'{self.__config_name}: '
        res+= self.__unique_name
        if display_params: res += '\n' + suff + ' -' + ', '.join([f'{k}: {v}' for k,v in self.__params.items() if not isinstance(v, dict)]) 
        res += '\n'
        for child in self.__children.values():
            if full_depth or isinstance(child, ConfiguredModule):
                res += child._name_structure(suff+'\t', include_config_name, display_params, full_depth)
        return res

class ConfiguredModule(ConfigurationNode):

    def __init__(self, parent=None, config_name=None, unique_name=None, params=None, **kwargs):
        if parent is None:
            if config_name is None: 
                config_name = type(self).__name__
                if unique_name is None: unique_name = config_name +'_0'
            elif unique_name is None: unique_name = config_name

        if params is None:
            params = {}
        params.update(kwargs)
        super().__init__(parent, config_name, unique_name, params)
        #self._param_init(*args, **kwargs)

    def get(self, ctor, name, *args, type_name=None, **kwargs):
        'Creates or gathers object which is initialized by the autoConfig.'
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = self.confiugre(ctor, *args, type_name, name, **kwargs)
        return self._modules[name]

    def _param_init(self, *args, **kwargs):
        'This method can be overwritten in order to specify specif parameters from other for a module.'
        pass