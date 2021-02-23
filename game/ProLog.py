from typing import List

from game import Action, AbstractGame
from gym_wrappers import ScalingObservationWrapper

class CartPole(AbstractGame):
    def __init__(self,problem):
        "Ide egy sima init kell. A problem egy adott probléma. Még nincs típusa. Lehet fájlnév vagy más. "

    @property
    def action_space_size(self)->int:
        "Ez végül nem kell, de itt hagyom emlékesztőtőül. (Eml_1)"
    
    def step(self,action):
        "Végre hajtja a lépést. Csak a végén van reward, ugye?"
        "Az action a /MuZero/game/game.py-ban lévő Action class egy példánya. "
        "Ott szimplán egy index. Ha máshogy kényelmes, akkor csak a __hash__, __init__, __eq__, __gt__ -ket kell tudnia."

    def terminal(self)->bool:
        "Találtunk-e megoldást?"

    def legal_actions(self)->List[Action]:
        "Visszaadja egy adott álapotban lévő lehetséges lépéseket."
    
    def make_image(self,state_index):
        "Visszaadja az adott állapotot. Ez az amit a GNN-nek kell kezelnie."
        "leancop_ml/montecarlo.py/row:174 helyen lévő data formátumú kellene."

    