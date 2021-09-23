from graph_data import *

data={
    'nodes_input_1': {
        'lens' :    [],
        'symbols' : [],
        'nodes' :   [],
        'sgn' :     [],
    },
    'nodes_input_2': {
        'lens' :    [],
        'symbols' : [],
        'nodes' :   [],
        'sgn' :     [],
    },
    'nodes_input_3': {
        'lens' :    [],
        'symbols' : [],
        'nodes' :   [],
        'sgn' :     [],
    },
    'symbol_inputs': {
        'lens' :    [],
        'nodes' :   [],
        'sgn' :     [],
    },
    'node_c_inputs': {
        'lens' :    [],
        'data' :    [],
    },
    'clause_inputs': {
        'lens' :    [],
        'data' :    [],
    },
    'ini_nodes' :   [],
    'ini_symbols':  [],
    'ini_clauses':  [],
    'num_nodes':    [],
    'num_symbols':  [],
    'num_clauses':  [],
}

gdata=GraphData(data)
print(gdata)
print(gdata.to_dict())