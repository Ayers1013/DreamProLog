from Env_ProLog import ProLog
from gnn.gnn_util import input2graph

env=ProLog()

env.reset()
state=env.gnnInput
data=input2graph(env.prolog, state)
str_data=[]
str_data.append(str(data))

print(str_data)

