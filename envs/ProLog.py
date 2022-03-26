import threading
import sys
import pathlib

import pyswip
import numpy as np
import tensorflow as tf

from misc import ConfiguredModule
from gnn import GraphData, extractActions, exctractImage, input2actionGraph, input2graph, gnn_output_sign

class ProblemLibrary:
    def __init__(self, config=None):
        #self.problem=lambda: "leancop/robinson_1p1__2.p"
        #self.problem=lambda: "leancop/pelletier21.p"
        directory="leancop/theorems/m2n140"
        self._load(directory)
        #print(f'Found {self.total} problem files.')
        #self.problem=lambda: "/".join(str(self.problems[np.random.randint(self.total)]).split("\\"))
        #self.problem=lambda: "leancop/pelletier21.p"

    def problem(self):
        #return "leancop/pelletier21.p"
        return  "/".join(str(self.problems[np.random.randint(self.total)]).split("\\"))

    def _load(self, directory):
        directory = pathlib.Path(directory).expanduser()
        self.problems=[]
        for filename in reversed(sorted(directory.glob('*.p'))):
            self.problems.append(filename)
        self.total=len(self.problems)

    def get(self):
        return self.problem()

class ProLog(ConfiguredModule):
    #LOCK=threading.Lock()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        '''
        This environment is not configurable at the moment.
        '''
        self.image_gnn = False
        self.image_axiom_mask = False
        self.image_action_mask = True
        self.image_text = True
        self.image_features = False

        'This can be either axiom or action. Does not change the behaior of the environment.'
        self.step_base = 'actions_mask'

        self.step_limit=25
        self.steps=0

        self.step_reward = 0.
        self.success_reward = 1
        self.failure_reward = -0.2
        self.invalid_reward = -1
        self.step_limit_reward = -0.5
        
        self.problems=ProblemLibrary()
        
        #with self.LOCK:
        self.prolog = pyswip.Prolog()
        self.prolog.consult("leancop/leancop_step.pl")
        # self.settings = "[conj, nodef, verbose, print_proof]"
        self.settings = "[conj, nodef, eager_reduction(1)]"
        problem=self.problems.get()
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, TextFeatures, TextActions, ActionsMask, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        self.simple_features = result["SimpleFeatures"]
        self.text_features = result["TextFeatures"]
        self.text_actions = result["TextActions"]
        self.actions_mask = result["ActionsMask"]

        self.ext_action_size = len(self.gnnInput[4])
        #TODO action_perm is redundant
        self.action_perm = self.gnnInput[5]

    @property
    def action_space_size(self)->int:
        return self.ext_action_size
    
    def step(self,action):
        self.steps+=1

        if self.step_base == 'action_mask':
            action = self.action_mask[action]
        elif self.step_base == 'axiom_mask':
            if(self.gnnInput[4][action]==0):
                action=-1
            elif(action==0):
                action=0
            else:
                action=np.array(self.gnnInput[4][:action]).sum()
        
        query = 'step_python({}, GnnInput, SimpleFeatures, TextFeatures, TextActions, ActionsMask, Result)'.format(action)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        #print(result)
        if len(result) == 0:
            self.result=-1
            reward = self.invalid_reward
        else:
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
            self.simple_features = result[0]["SimpleFeatures"]
            self.text_features = result[0]["TextFeatures"]
            self.text_actions = result[0]["TextActions"]
            self.actions_mask = result[0]["ActionsMask"]
            self.action_perm = self.gnnInput[5]
            reward=self.reward()
            
        obs = self.image()
        #print('Observation meta:', len(obs['axiom_mask']), obs['action_space']['num_clauses'], obs['action_space']['num_nodes'])
        return (obs, reward, self.terminal(), {}) 
    
    def reset(self, problem=None):
        self.steps=0
        
        if problem==None:
            problem=self.problems.get()
        #remove out '.p' and /
        self.current_problem="__".join(problem[:-2].split('/')[2:])
        #print('Loaded problem:', self.current_problem)
        query = 'init_python("{}",{},GnnInput, SimpleFeatures, TextFeatures, TextActions, ActionsMask, Result)'.format(problem, self.settings)
        #print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]
        #result[0] -> result
        self.simple_features = result["SimpleFeatures"]
        self.text_features = result["TextFeatures"]
        self.text_actions = result["TextActions"]
        self.actions_mask = result["ActionsMask"]

        self.ext_action_size = len(self.gnnInput[4])
        self.action_perm = self.gnnInput[5]

        obs = self.image()
        #print('Observation meta:', len(obs['axiom_mask']), obs['action_space']['num_clauses'], obs['action_space']['num_nodes'])
        return obs

    def image(self):
        image={}

        if self.image_features:
            image['features'] = np.tanh(np.array(self.simple_features,np.float32)*0.1)
        if self.image_axiom_mask:
            image['axiom_mask'] = self.gnnInput[4]
        if self.image_action_mask:
            image['action_mask'] = self.actions_mask
        if self.image_gnn:
            image['gnn']=exctractImage(self.prolog, self.gnnInput)
            image['action_space']=extractActions(self.prolog, self.gnnInput)
        if self.image_text:
            image['text'] = self.text_features
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

    def output_sign(self, batch_size=0, batch_length=None):
        _shape=(batch_size, batch_length) if batch_size!=0 else (batch_length,)

        spec=lambda x ,dt: tf.TensorSpec(shape=_shape+x, dtype=dt)
        sign={
            'image': spec((14,), tf.float32),
            'features': spec((14,), tf.float32),
            'axiom_mask': spec((None,), tf.int32)
            }

        include_num=True

        if self.image_gnn:
            sign['gnn']=gnn_output_sign(lambda x: tf.RaggedTensorSpec(shape=_shape+x, dtype=tf.int32), include_num)
            sign['action_space']=gnn_output_sign(lambda x: tf.TensorSpec(shape=()+x, dtype=tf.int32), include_num)

        return sign


    def viz_state(self):
        print("--------------------")
        print("Open goals:")
        for g in self.text_features:
            print("   G:", g)

        print("All actions: ")
        for a in self.text_actions:
            print("  AA:", a)

        print("Action mask: ")
        print(self.actions_mask)

def meta_data(problem_name, image_gnn = False, image_text = True):
    env=ProLog()
    env.reset("leancop/theorems/"+problem_name)
    image = {}

    if image_gnn:
        image['action_space_gnn']=extractActions(env.prolog, env.gnnInput)
    if image_text:
        image['action_space_text']=env.text_actions
    return image
