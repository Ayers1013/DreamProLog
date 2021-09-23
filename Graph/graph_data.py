import tensorflow as tf
import numpy as np

class DataSkeleton:

    def __init__(self, data):
        for attr in self.__slots__:
            #assert attr in data
            setattr(self, attr, data[attr])

    def copy(self):
        copy=self.__class__()
        set_fn=lambda attr, data: setattr(copy, attr, data)
        for attr in self.__slots__:
            data=getattr(self, attr)
            if isinstance(data, DataSkeleton) or isinstance(data, np.ndarray):
                set_fn(attr, data.copy())
            else: 
                set_fn(attr, data)
    
    def to_dict(self):
        dct = {}
        for attr in self.__slots__:
            x=getattr(self, attr)
            if isinstance(x, DataSkeleton):
                dct[attr] = x.to_dict()
            #We use the fact every element of a tuple is an instance of DataSkeleton
            elif isinstance(x, list):
                for i, y in enumerate(x):
                    dct[attr+f'_{i+1}']=y.to_dict()
            
            else:
                dct[attr]=x
        return dct
        

class GraphHyperEdgesA(DataSkeleton):
    __slots__ = ['lens', 'symbols', 'nodes', 'sgn']
    def __init__(self, data):
        super().__init__(data)

class GraphHyperEdgesB(DataSkeleton):
    __slots__ = ['lens', 'nodes', 'sgn']
    def __init__(self, data):
        super().__init__(data)

class GraphEdges(DataSkeleton):
    __slots__ = ['lens', 'data']
    def __init__(self, data):
        super().__init__(data)

class GraphData(DataSkeleton):
    __slots__ = ['node_inputs', 'symbol_inputs',
                 'node_c_inputs', 'clause_inputs',
                 'ini_nodes', 'ini_symbols', 'ini_clauses',
                 'num_nodes', 'num_symbols', 'num_clauses']
    def __init__(self, data):
        self.node_inputs=[GraphHyperEdgesA(data[f'node_inputs_{i}']) for i in range(1,4)]
        self.symbol_inputs=GraphHyperEdgesB(data['symbol_inputs'])
        self.node_c_inputs=GraphEdges(data['node_c_inputs'])
        self.clause_inputs=GraphEdges(data['clause_inputs'])
        for attr in [
            'ini_nodes', 'ini_symbols', 'ini_clauses',
            'num_nodes', 'num_symbols', 'num_clauses']:
            setattr(self, attr, data[attr])

    


