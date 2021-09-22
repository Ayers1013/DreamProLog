from _typeshed import Self
import numpy as np
import tensorflow as tf 

class Policy:
    def __init__(self, cursor, actions):
        self.actions=actions
        self.cursor=0

    def __call__(self, *args):
        return self

    def sample(self):
        action=self.actions[self.cursor]
        self.cursor+=1
        return action

    def mode(self):
        action=self.actions[self.cursor]
        self.cursor+=1
        return action

class Judge:
    def __init__(self, agent):
        self.agent=agent

    def simulate_trajectory(self, start, episode):
        policy=Policy(start, episode['actions'])
        state=tf.nest.map_structure(lambda x: x[start], episode)
        self.agent._task_behavior._imagine(state, policy, len(episode['actions']-start))
        
        