import pyswip
import threading
import sys

#import gym
import numpy as np

class ActionSpace:
    def __init__(self, _n):
        self.n=_n

    def sample(self):
        arr=np.zeros(self.n)
        arr[np.random.randint(self.n)]=1.0
        #BUG#002
        return arr
        #return np.random.randint(self.n)

    @property
    def shape(self):
        return self.n

class ProLog:
    LOCK=threading.Lock()
    
    def __init__(self,problems=None):
        #problems is a generator function
        self.step_limit=10
        self.steps=0
        if problems==None:
            def gen():
                while True:
                    yield "leancop/robinson_1p1__2.p"
                    #yield "leancop/pelletier21.p"
            self.problems=gen()
        else:
            self.problems=problems

        self.small_reward = 0.01 # TODO
        
        with self.LOCK:
            self.prolog = pyswip.Prolog()
        self.prolog.consult("leancop/leancop_step.pl")
        # self.settings = "[conj, nodef, verbose, print_proof]"
        self.settings = "[conj, nodef]"
        problem=next(self.problems)
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
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
        #TODO Correct this!
        return ActionSpace(self.ext_action_size)

    @property
    def action_space_size(self)->int:
        return self.ext_action_size
    
    def step(self,action):
        self.steps+=1
        #TODO output obs,reward,done, info
        #TODO BUG#001
        if(action.shape!=(1)):
            action=np.argmax(action)
        query = 'step_python({}, GnnInput, SimpleFeatures, Result)'.format(action)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        if len(result) == 0:
            self.result=-1
            reward = -self.small_reward#-1 
        else:
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
            self.simple_features = result[0]["SimpleFeatures"]
            if self.result == -1:
                reward = -self.small_reward
            elif self.result == 1:
                reward =1#self.small_reward
            else:
                if(self.steps==self.step_limit):
                    self.result=-1
                    reward=-1
                else:
                    reward = 0

        #return ({'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
        return ({'image':np.ones(16)*self.steps*0.1},#'features': self.get_features()},
            np.float64(reward), 
            self.terminal,
            {}) 
    
    def reset(self):
        self.steps=0
        #with self.LOCK:
        #    self.prolog=pyswip.Prolog()

        
        # self.prolog.consult("leancop/leancop_step.pl") # TODO I think we don't need to reconsult
        problem=next(self.problems)
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]        
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        #result[0] -> result
        self.simple_features = result["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])

        #return {'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
        return {'image':np.zeros(16)}#, 'features': self.get_features()}


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
