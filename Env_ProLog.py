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

        action_size=len(self.gnnInput[4])
        action_space=np.zeros((action_size,32))
        action_space[np.arange(action_size),np.arange(action_size)]=1.

        return (
            {'image':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'features':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'action_space':action_space},
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

        action_size=len(self.gnnInput[4])
        action_space=np.zeros((action_size,32))
        action_space[np.arange(action_size),np.arange(action_size)]=1.

        #return {'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
        #return {'image':np.zeros(16)}
        return {'image':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'features':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'action_space':action_space}


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
            reward=100.0
            self.done=True
        elif(self.pos_x<0 or self.pos_y<0 or self.pos_x>= self.board_size or self.pos_y>= self.board_size):
            reward=-1.0
            self.done=True
        else:
            reward=0.1*(10-self.pos_x-self.pos_y)**2
        if(not self.done and self.steps>=self.step_limit):
            reward=-1.0
            self.done=True

        
        image=np.zeros(self.board_size**2)
        dist=lambda x,y: abs(self.pos_x-x)+abs(self.pos_y-y)
        _lambda=np.log(0.7)
        if(not self.done):
            #Trivial
            #image[self.pos_x+self.pos_y*self.board_size]=1.0
            for x in range(self.board_size):
                for y in range(self.board_size):
                    image[x+y*self.board_size]=dist(x,y)
            image=np.exp(_lambda*image)

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
