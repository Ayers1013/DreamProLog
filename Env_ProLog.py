import pyswip
import threading

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
        settings = "[conj, nodef, verbose, print_proof]"
        problem=next(self.problems)
        query = 'init("{}",{},state(Tableau, Actions, Result)), state2gnnInput(state(Tableau, Actions, Result),GnnInput)'.format(problem, settings)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]        
        self.tableau = result["Tableau"]
        self.actions = result["Actions"]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]

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
        query = 'step({},state(Tableau, Actions, Result)), writeln(result-Result), !, state2gnnInput(state(Tableau, Actions, Result),GnnInput)'.format(action)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        if len(result) == 0:
            self.tableau = "failure"
            self.actions = []
            self.result=-1
            reward = -1
        else:
            print(result[0])
            self.tableau = result[0]["Tableau"]
            self.actions = result[0]["Actions"]
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
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

        
        self.prolog.consult("leancop/leancop_step.pl")
        settings = "[conj, nodef, verbose, print_proof]"
        problem=next(self.problems)
        query = 'init("{}",{},state(Tableau, Actions, Result)), state2gnnInput(state(Tableau, Actions, Result),GnnInput)'.format(problem, settings)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]        
        self.tableau = result["Tableau"]
        self.actions = result["Actions"]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]

        self.ext_action_size = len(self.gnnInput[4])

        return self.make_image()
        

    @property
    def terminal(self)->bool:
        return self.result != 0

    def legal_actions(self):
        return self.actions # TODO agree on adequate format
    
    def make_image(self):
        return self.gnnInput

    def get_features(self):
        #Returns the features 
        return None
