from typing import List

#from game import Action#, AbstractGame
#from gym_wrappers import ScalingObservationWrapper
import pyswip

class LeanCoP:
    def __init__(self,problem):

        self.small_reward = 0.01 # TODO
        
        self.prolog = pyswip.Prolog()
        self.prolog.consult("leancop/leancop_step.pl")
        settings = "[conj, nodef, verbose, print_proof]"
        query = 'init("{}",{},state(Tableau, Actions, Result)), state2gnnInput(state(Tableau, Actions, Result),GnnInput)'.format(problem, settings)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))[0]        
        self.tableau = result["Tableau"]
        self.actions = result["Actions"]
        self.result = result["Result"]
        self.gnnInput = result["GnnInput"]

        self.ext_action_size = len(self.gnnInput[4])


    @property
    def action_space_size(self)->int:
        "Ez végül nem kell, de itt hagyom emlékesztőtőül. (Eml_1)"
        return self.ext_action_size
    
    def step(self,action):
        "Végre hajtja a lépést. Csak a végén van reward, ugye?"
        "Az action a /MuZero/game/game.py-ban lévő Action class egy példánya. "
        "Ott szimplán egy index. Ha máshogy kényelmes, akkor csak a __hash__, __init__, __eq__, __gt__ -ket kell tudnia."

        query = 'step({},state(Tableau, Actions, Result)), writeln(result-Result), !, state2gnnInput(state(Tableau, Actions, Result),GnnInput)'.format(action)
        print("Query:\n   ", query, "\n")
        result = list(self.prolog.query(query))
        if len(result) == 0:
            self.tableau = "failure"
            self.actions = []
            self.result=-1
            return -1
        else:
            print(result[0])
            self.tableau = result[0]["Tableau"]
            self.actions = result[0]["Actions"]
            self.result = result[0]["Result"]
            self.gnnInput = result[0]["GnnInput"]
            if self.result == -1:
                return -self.small_reward
            elif self.result == 1:
                return self.small_reward
            else:
                return 0

    def terminal(self)->bool:
        "Találtunk-e megoldást?"
        return a.result != 0

    def legal_actions(self):
        "Visszaadja egy adott álapotban lévő lehetséges lépéseket."
        return self.actions # TODO agree on adequate format
    
    def make_image(self,state_index):
        "Visszaadja az adott állapotot. Ez az amit a GNN-nek kell kezelnie."
        "leancop_ml/montecarlo.py/row:174 helyen lévő data formátumú kellene."
        return self.gnnInput


problem = "leancop/pelletier21.p"
a = LeanCoP(problem)
# a.step(0)
