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
        # arr[np.random.randint(4)]=1.0
        #BUG#002
        return arr
        #return np.random.randint(self.n)

    @property
    def shape(self):
        return self.n

def new_dummy_state(state,action):
    arr=np.array([[0.77258772, 0.75418344, 0.8259575 , 0.02423857, 0.86239288,
        0.33258206, 0.95879772, 0.47319562, 0.31980994, 0.13325199,
        0.66628455, 0.80085589, 0.40491547, 0.80875495, 0.61982221,
        0.08043198],
       [0.82731647, 0.29043499, 0.30640893, 0.28646636, 0.80434398,
        0.55883647, 0.25915326, 0.51088844, 0.86824613, 0.53609049,
        0.15376353, 0.4201172 , 0.12115401, 0.75851432, 0.81218553,
        0.59679522],
       [0.19487862, 0.44658057, 0.3547951 , 0.55726615, 0.60476992,
        0.65753579, 0.55189446, 0.06121563, 0.15233983, 0.71434028,
        0.87651087, 0.22062229, 0.56857382, 0.14328761, 0.15237773,
        0.45713311],
       [0.52926322, 0.49737374, 0.08980527, 0.74294145, 0.75640947,
        0.87127879, 0.28077583, 0.31190462, 0.10956359, 0.33175913,
        0.38043219, 0.67974034, 0.46698045, 0.03812514, 0.88126502,
        0.83942109],
       [0.57589548, 0.65682949, 0.01000789, 0.47975863, 0.56174634,
        0.65936929, 0.89641352, 0.66110823, 0.86270629, 0.69519311,
        0.28986063, 0.38407464, 0.51005799, 0.11880446, 0.29196281,
        0.42781268],
       [0.3034209 , 0.86756469, 0.3674309 , 0.19065117, 0.91316337,
        0.19388465, 0.14741657, 0.13626633, 0.61207659, 0.84130062,
        0.9217264 , 0.24627352, 0.50803292, 0.16054827, 0.58236423,
        0.64465774],
       [0.07178429, 0.27736165, 0.98123672, 0.7697868 , 0.01396827,
        0.13451188, 0.34998996, 0.56483173, 0.40838422, 0.47793902,
        0.27818172, 0.42070331, 0.044446  , 0.00568323, 0.96709433,
        0.75556281],
       [0.27999107, 0.29434092, 0.26028494, 0.70457187, 0.3866507 ,
        0.28851239, 0.06459557, 0.62045259, 0.65207226, 0.74040966,
        0.54382434, 0.00838744, 0.6702245 , 0.39741768, 0.88180169,
        0.60960996],
       [0.6728838 , 0.49456642, 0.44253753, 0.7321289 , 0.28238044,
        0.83537702, 0.12988673, 0.30499548, 0.03994815, 0.30825163,
        0.43058451, 0.55346695, 0.70113272, 0.79252324, 0.80364323,
        0.41036875]])
    return state+arr[action]

class ProLog:
    LOCK=threading.Lock()
    
    def __init__(self,problems=None):
        #problems is a generator function
        self.step_limit=10
        self.steps=0
        if problems==None:
            def gen():
                while True:
                    #yield "leancop/robinson_1p1__2.p"
                    yield "leancop/pelletier21.p"
            self.problems=gen()
        else:
            self.problems=problems

        self.step_reward = 0.3#0.01
        self.success_reward = 1
        self.failure_reward = -0.1
        self.invalid_reward = -1
        self.step_limit_reward = -0.5
        
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
        self.action_perm = self.gnnInput[5]

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

        """
        indices = np.where(self.action_perm==action)[0]
        # print(indices)
        if len(indices) > 0:
            action = indices[0]
        else:
            action = -1
        """
        #print(action)
        if(self.gnnInput[4][action]==0):
            action=-1
        elif(action==0):
            action=0
        else:
            action=np.array(self.gnnInput[4][:action]).sum()
    
        
        #action=0
        query = 'step_python({}, GnnInput, SimpleFeatures, Result)'.format(action)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        if len(result) == 0:
            self.result=-1
            reward = self.invalid_reward
        else:
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
            self.simple_features = result[0]["SimpleFeatures"]
            self.action_perm = self.gnnInput[5]
            if self.result == -1:
                reward = self.failure_reward
            elif self.result == 1:
                reward = self.success_reward
            else:
                if(self.steps==self.step_limit):
                    self.result=-1
                    reward = self.step_limit_reward
                else:
                    reward = self.step_reward

        #return ({'image':self.gnnInput, 'ram': None, 'features': self.get_features()},


        return (
            {'image':np.tanh(np.array(self.simple_features,np.float32)*0.1)},
            #{'image':np.ones(16)*self.steps*0.1},#'features': self.get_features()},
            np.float64(reward), 
            self.terminal,
            {}) 
    
    def reset(self):
        #print("_reset_")
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
        self.action_perm = self.gnnInput[5]

        #return {'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
        #return {'image':np.zeros(16)}
        return {'image':np.tanh(np.array(self.simple_features,np.float32)*0.1)}


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


class DummyEnv:
    LOCK=threading.Lock()
    
    def __init__(self,problems=None):
        #problems is a generator function
        self.step_limit=25
        self.board_size=5
        self.steps=0
        self.pos_x=2
        self.pos_y=2
        self.done=False

    @property
    def observation_space(self):
        #TODO
        return 0

    @property
    def action_space(self):
        #TODO Correct this!
        return ActionSpace(4)

    @property
    def action_space_size(self)->int:
        return 4
    
    def step(self,action):
        self.steps+=1

        #TODO output obs,reward,done, info
        #TODO BUG#001
        if(action.shape!=(1)):
            action=np.argmax(action)
        
        reward=0.0

        if(action==0):
            self.pos_x+=1
        elif(action==1):
            self.pos_x-=1
            reward=0.1
        elif(action==2):
            self.pos_y+=1
        else:
            self.pos_y-=1
            reward=0.1

        if(self.pos_x==0 and self.pos_y==0):
            reward=1.0
            self.done=True
        elif(self.pos_x<0 or self.pos_y<0 or self.pos_x>= self.board_size or self.pos_y>= self.board_size):
            reward=-1.0
            self.done=True

        if(not self.done and self.steps>=self.step_limit):
            reward=-1.0
            self.done=True

        image=np.zeros(self.board_size**2)
        if(not self.done):
            image[self.pos_x+self.pos_y*self.board_size]=1.0

        return (
            {'image':image},
            np.float64(reward), 
            self.done,
            {}) 
    
    def reset(self):
        #problems is a generator function
        self.step_limit=25
        self.board_size=5
        self.steps=0
        self.pos_x=2
        self.pos_y=2
        self.done=False
        return {'image':np.zeros(self.board_size**2)}
