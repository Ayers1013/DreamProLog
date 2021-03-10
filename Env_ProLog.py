import pyswip
import threading
import sys

#import gym
import numpy as np
class ProLog:
    LOCK=threading.Lock()
    
    def __init__(self,problems=None):
        #problems is a generator function
        if problems==None:
            def gen():
                while True:
                    yield "leancop/pelletier21.p"
            self.problems=gen()
        else:
            self.problems=problems

        self.small_reward = 0.01 # TODO
        
        with self.LOCK:
            self.prolog = pyswip.Prolog()
        self.prolog.consult("leancop/leancop_step.pl")
        self.settings = "[conj, nodef, verbose, print_proof]"
        problem=next(self.problems)
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        self.simple_features = result["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])

    @property
    def observation_space(self):
        #TODO
        return 0

    @property
    def action_space(self):
        #TODO
        return 0

    @property
    def action_space_size(self)->int:
        return self.ext_action_size
    
    def step(self,action):
        #TODO output obs,reward,done, info
        query = 'step_python({}, GnnInput, SimpleFeatures, Result)'.format(action)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        if len(result) == 0:
            self.result=-1
            reward = -1
        else:
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
            self.simple_features = result[0]["SimpleFeatures"]
            if self.result == -1:
                reward = -self.small_reward
            elif self.result == 1:
                reward = self.small_reward
            else:
                reward = 0

        return ({'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
            reward, 
            self.terminal,
            None) 
    
    def reset(self):
        with self.LOCK:
            self.prolog=pyswip.Prolog()

        
        # self.prolog.consult("leancop/leancop_step.pl") # TODO I think we don't need to reconsult
        problem=next(self.problems)
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]        
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        self.simple_features = result[0]["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])

        return self.make_image()
        

    @property
    def terminal(self)->bool:
        return self.result != 0

    def legal_actions(self):
        return None # TODO extract from self.gnnInput
    
    def make_image(self):
        return self.gnnInput

    def get_features(self):
        #Returns the features 
        return self.simple_features
