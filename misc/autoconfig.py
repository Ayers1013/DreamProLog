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
        if isinstance(config, str):
            _, ext = config.split('.')
            if ext == 'yaml':
                self.load_from_yaml(config)
        
        elif isinstance(config, list):
            for c in config:
                self.load(c)


    def load_from_yaml(self, path):
        self.__params.update(yaml.safe_load((pathlib.Path(path)).read_text()))


class ConfigurationNode:
    __param_prefix = ''
    def __init__(self, parent=None, config_name=None, unique_name=None, param_prefix=None, params=None):
        'Tree structure:'
        super().__init__()
        self.__parent = parent
        self.__core = parent.__core if parent is not None else self
        self.__children = {}
        self.__config_name = config_name
        self.__unique_name = unique_name
        self.__param_prefix = param_prefix or ''

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

    # TODO works well but a better defined behavior is necessary
    def __getattr__(self, name):
        try: super().__getattr__(name)
        except:
            length = len(self.__param_prefix)
            if name == '_ConfigurationNode__params': raise AttributeError('ConfiguredModule must be initialized. Just add super().__init__()')
            if name[:length] == self.__param_prefix: name = name[length:]
            if GET_PARAMETER_BY_ATTR and name[0] != '_':
                    return self[name]
            raise AttributeError(f'The {name} attribute is not found.')

    def __getitem__(self, name):
        'Gather parameters by name.'
        node = self
        while node is not None:
            if name in node.__params:
                return node.__params[name]
            node = node.__parent
        raise AttributeError(f'{name}')

    def __get_param(self, name):
        node = self
        while node is not None:
            if name in node.__params:
                return node.__params[name]
            node = node.__parent
        return None

    def __iter__(self):
        iter_pointer = []
        current_node = self
        while True:
            yield current_node
            if len(current_node.__children)>0:
                iter_pointer.append(iter(current_node.__children.values()))
            while len(iter_pointer)>0:
                try: 
                    current_node = next(iter_pointer[-1])
                    break
                except StopIteration:
                    iter_pointer.pop()
            if len(iter_pointer) == 0:
                break

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
                # TODO:ow_order decide about the order
                param = self.__get_param(k)
                if k in kwargs: param = kwargs[k]
                if param is not None: ckwargs[k] = param
            return ctor(*args, **ckwargs)

    def _stats(self):
        return (self.__config_name, self.__unique_name, self.__params, self.__parent)

    def _name_structure(self, prefix = '', include_config_name = True, display_params = True, full_depth = True):
        res = prefix
        if include_config_name: res+= f'{self.__config_name}: '
        res+= self.__unique_name
        if display_params: res += '\n' + prefix + ' -' + ', '.join([f'{k}: {v}' for k,v in self.__params.items() if not isinstance(v, dict)]) 
        res += '\n'
        for child in self.__children.values():
            if full_depth or isinstance(child, ConfiguredModule):
                res += child._name_structure(prefix+'\t', include_config_name, display_params, full_depth)
        return res

class ConfiguredModule(ConfigurationNode):

    def __init__(self, *args, parent=None, config_name=None, unique_name=None, param_prefix=None, params=None, **kwargs):
        if parent is None:
            if config_name is None: 
                config_name = type(self).__name__
                if unique_name is None: unique_name = config_name +'_0'
            elif unique_name is None: unique_name = config_name

        if params is None: params = {}
        params.update(kwargs)
        super().__init__(parent, config_name, unique_name, param_prefix, params)
        self._param_init(*args, params=self._ConfigurationNode__params)

    def get(self, ctor, name, *args, type_name=None, **kwargs):
        'Creates or gathers object which is initialized by the autoConfig.'
        if not hasattr(self, '_modules'):
            self._modules = {}
        if name not in self._modules:
            self._modules[name] = self.confiugre(ctor, *args, type_name, name, **kwargs)
        return self._modules[name]

    def _param_init(self, *args, params):
        'This method can be overwritten in order to specify specif parameters from other for a module.'

        # Collects default parameters
        for k, v in self._param_default.items():
            if k not in params: params[k] = v
        # Collects positional arguments.
        for k, v in zip(self._param_args, args):
            params[k] = v

    @property
    def _param_default(self):
        'This method can be overwritten in order to specify default parameters.'
        return {}
    
    @property
    def _param_args(self):
        'This method can be overwritten to allow the Module to be initialised by positional arguments. It returns a list with the name of positional arguments.'
        return []