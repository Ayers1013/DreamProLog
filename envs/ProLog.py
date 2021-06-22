import pyswip
import threading
import sys

#graph_data test
from gnn.graph_data import GraphData
data=GraphData()
#data.load_from_str("1 1 1 1 1 1 1 1 0 1 0 1,0 1 2 3 4 3 1 0 3 3,-1 -1 -1 -1 -1 -1 2 -1 -1 -1 4 -1 -1 -1 -1 -1 8 -1 10 -1,-1 1 1 -1 1 1 -1 1 1 -1;0 0 1 0 1 0 0 0 1 0 1 0,3 3 3 3,3 -1 5 -1 9 -1 11 -1,-1 1 1 -1;0 0 0 0 0 0 0 0 0 0 0 0,,,;2 2 1 4 1,0 -1 -1 7 -1 -1 1 -1 -1 6 -1 -1 2 -1 -1 3 2 -1 5 4 -1 9 8 -1 11 10 -1 4 -1 -1,-1 1 1 -1 1 -1 1 1 -1 1;2 2 0 1 0 1 2 1 0 1 0 1,0 1 2 4 2 3 3 5 4 4 5;1 1 2 2 3 2,0 0 1 3 5 6 7 1 9 6 11;0 0 1 0 1 0 0 0 3 0 3 0;1 1 0 1 0;0 1 3 3 3 3;0 0 0 0 0 0 1 0 0 0 0")
data.load_from_str('1 1 1 1 1 1 1 0 1 1 0 1 1 0 0 1 1 1 1 1 0 1 1 0 0 1 1 1 1 1 0 1 0 1 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1,0 0 1 2 3 2 4 2 4 3 4 2 3 3 2 4 5 4 2 5 5 3 4 2 2 4 4 4 4 4 4 4 4 5 5 4 4 4 4 2 2 4 3 3 4 4 4,-1 -1 -1 -1 -1 -1 2 -1 3 3 3 -1 4 5 7 -1 2 8 10 2 11 10 14 -1 13 15 13 14 17 -1 16 18 20 2 21 2 24 -1 23 25 23 24 27 23 26 28 30 -1 32 -1 31 33 30 32 36 36 38 39 39 38 42 43 42 45 45 43 48 49 51 52 50 53 48 51 49 52 57 58 57 -1 58 -1 60 61 63 64 66 67 65 68 63 66 64 67,-1 1 1 1 1 1 1 1 1 1 -1 1 1 1 1 -1 1 -1 1 1 1 1 -1 1 1 1 -1 -1 1 -1 -1 1 1 1 1 -1 1 1 1 1 1 -1 1 1 -1 1 1;0 0 2 2 1 0 0 1 0 0 1 1 0 2 1 0 1 1 0 0 1 1 0 2 1 0 1 1 0 0 2 1 1 0 0 0 1 0 1 1 0 0 2 0 0 1 0 0 2 1 1 1 0 0 0 0 0 2 1 0 1 0 0 2 1 1 1 0 0 0 0 0,2 4 3 2 4 2 3 4 3 3 2 4 2 5 4 5 5 2 4 3 2 4 4 2 4 4 4 4 4 4 5 4 4 4 5 4 2 2 4 3 4 4 4 3,3 -1 9 8 4 3 5 -1 6 5 8 -1 11 2 12 10 16 15 17 14 15 -1 19 18 18 -1 21 2 22 2 26 25 27 24 25 -1 29 28 28 23 31 -1 35 32 34 33 33 -1 37 36 40 39 41 38 44 43 46 45 47 43 50 49 55 51 56 52 54 53 53 52 59 58 60 -1 61 -1 62 61 65 64 70 66 71 67 69 68 68 67,1 1 1 1 1 1 1 -1 1 1 1 -1 1 1 -1 1 1 1 -1 1 1 -1 1 1 -1 1 -1 -1 1 1 1 1 1 -1 1 1 1 1 -1 1 1 1 -1 1;0 0 3 1 0 1 0 0 1 0 1 0 0 0 1 1 0 0 1 0 0 0 0 1 1 1 0 0 1 0 0 0 1 1 0 0 1 0 1 1 0 0 0 2 0 1 0 0 0 1 0 1 2 1 0 0 0 0 1 0 0 1 0 0 1 0 1 2 1 0 0 0,3 5 4 3 4 4 4 3 3 4 3 5 5 4 4 4 4 4 4 4 4 4 5 4 5 4 4 4 4 3 4 3 4 4,11 10 21 20 22 21 4 3 6 4 9 2 12 11 17 13 16 13 19 16 28 27 27 23 26 23 29 26 35 30 34 31 37 36 41 39 40 38 44 42 47 45 46 42 50 48 55 48 53 51 56 49 54 50 59 57 62 60 65 63 70 63 68 66 71 64 69 65,1 1 -1 1 1 1 -1 1 1 -1 1 1 1 -1 -1 1 -1 -1 1 -1 1 1 1 1 1 1 -1 1 -1 1 1 1 1 -1;2 1 10 7 22 5,0 -1 -1 1 -1 -1 2 -1 -1 3 2 -1 5 3 -1 8 7 -1 15 14 -1 18 17 -1 25 24 -1 31 30 -1 33 32 -1 60 57 -1 61 58 -1 4 3 3 11 10 2 16 13 15 17 13 14 28 27 23 65 63 64 68 66 67 6 4 5 9 2 8 12 11 10 19 16 18 22 21 2 29 26 28 34 31 33 35 30 32 37 36 36 40 38 39 41 39 38 44 42 43 46 42 45 47 45 43 54 50 53 55 48 51 56 49 52 59 57 58 62 60 61 69 65 68 70 63 66 71 64 67 21 20 2 26 23 25 27 23 24 50 48 49 53 51 52,-1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 1 -1 -1 1 -1 -1 1 1 -1 1 1 1 -1 -1 1 1 1 1 1 1 1;3 1 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 1 1 0 1 0 0 1 1 0 0 1 0 1 1 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 1 1 1,0 1 3 2 2 4 5 6 7 8 9 9 10 11 11 12 12 12 13 13 13 14 14 15 15 15;1 1 2 1 1 1 1 1 1 2 1 2 3 3 2 3,0 0 1 6 0 9 12 19 22 29 34 35 37 40 41 44 46 47 54 55 56 59 62 69 70 71;0 0 1 1 1 1 0 3 1 0 3 1 0 3 3 1 1 1 1 0 3 1 0 3 3 1 1 1 1 0 3 1 3 1 0 0 3 0 3 3 0 0 3 3 0 3 0 0 3 3 1 3 3 1 0 0 0 3 3 0 1 1 0 3 3 1 3 3 1 0 0 0;1 0 0 0 1 0;0 1 3 3 3 3 3 3 3 3 3 3 3 3 3 3;0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0')


#import gym
import numpy as np
import tensorflow as tf

class ProblemLibrary:
    def __init__(self, config=None):
        #self.problem="leancop/robinson_1p1__2.p"
        self.problem="leancop/pelletier21.p"
    def get(self):
        return self.problem

class ProLog:
    LOCK=threading.Lock()
    
    def __init__(self, gnn=True):
        #problems is a generator function
        self.gnn=gnn

        self.step_limit=10
        self.steps=0
        
        self.problems=ProblemLibrary()

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
        problem=self.problems.get()
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        self.simple_features = result["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])
        #TODO action_perm is redundant
        self.action_perm = self.gnnInput[5]

    @property
    def action_space_size(self)->int:
        return self.ext_action_size
    
    def step(self,action):
        self.steps+=1

        #TODO output obs,reward,done, info
        #TODO BUG#001
        if(action.shape!=(1)):
            action=np.argmax(action)

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
            reward=self.reward()
            
        
        return (self.image(), reward, self.terminal(), {}) 
    
    def reset(self):
        self.steps=0
        
        problem=self.problems.get()
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]        
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        #result[0] -> result
        self.simple_features = result["SimpleFeatures"]

        self.ext_action_size = len(self.gnnInput[4])
        self.action_perm = self.gnnInput[5]

        return self.image()

    def image(self):
        action_size=len(self.gnnInput[4])
        action_space=np.zeros((action_size,32))
        action_space[np.arange(action_size),np.arange(action_size)]=1.

        image={'image':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'features':np.tanh(np.array(self.simple_features,np.float32)*0.1),
            'action_space':action_space}

        if self.gnn:
            image['gnn']=data.convert_to_dict()

        #return {'image':self.gnnInput, 'ram': None, 'features': self.get_features()},
        #return {'image':np.zeros(16)}
        return image

    def terminal(self)->bool:
        return self.result != 0

    def reward(self):
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
        return np.float64(reward)
    
    @property
    def output_sign(self):
        outputs=[
            'node_inputs_1/lens',
            'node_inputs_1/symbols',
            'node_inputs_1/nodes',
            'node_inputs_1/sgn',
            'node_inputs_2/lens',
            'node_inputs_2/symbols',
            'node_inputs_2/nodes', 
            'node_inputs_2/sgn', 
            'node_inputs_3/lens', 
            'node_inputs_3/symbols', 
            'node_inputs_3/nodes', 
            'node_inputs_3/sgn', 
            'symbol_inputs/lens', 
            'symbol_inputs/nodes', 
            'symbol_inputs/sgn', 
            'node_c_inputs/lens', 
            'node_c_inputs/data', 
            'clause_inputs/lens', 
            'clause_inputs/data', 
            'ini_nodes', 
            'ini_symbols', 
            'ini_clauses',
            'num_nodes',
            'num_symbols',
            'num_clauses'
        ]
        gnnSpec={}
        for name in outputs:
            if name=='symbol_inputs/nodes': gnnSpec[name]=tf.TensorSpec(shape=(None,3), dtype=tf.int32)
            elif name.find("/")!=-1 and name.split("/")[1]=='nodes': gnnSpec[name]=tf.TensorSpec(shape=(None,2), dtype=tf.int32)
            else: gnnSpec[name]=tf.TensorSpec(shape=(None,), dtype=tf.int32)

        sign={
            'image': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            'features': tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            'action_space': tf.TensorSpec(shape=(None, None, 32), dtype=tf.float32)
            }

        if self.gnn:
            sign['gnn']=gnnSpec

        return sign


