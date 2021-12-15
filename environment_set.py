import tensorflow as tf
import numpy as np
from typing import List, Dict

from envs import make_env

class EnvStats:
    def __init__(self, names: List[str]):
        self._names = names
        self._stats: Dict[(str, str)] = {}


    def callback(self, episode):
        name = episode.current_problem

        self._stats[(name, 'count')]+=1
        self._stats[(name, 'elo')]+=0

    def save(self):
        pass

    def load(self):
        pass

class EnvironmentSet:
    def __init__(self, names: List[str],  config, callbacks):
        self._names = names

        self._envs = [make_env(config, callbacks) for _ in config.env_nums]
        self.envStats = EnvStats(self._names)


    def reset(self, name):
        #assert name in self._names
        for env in self._envs:
            env.reload_env(name)

    def save(self):
        pass

    def load(self):
        pass
