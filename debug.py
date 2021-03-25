from Env_ProLog import ProLog as Pl

a=Pl()
a.step(1)

from gnn.gnn_util import *
graph=TermGraph(PrologDecoder(a.prolog))

curr_lit, path, all_goals, ext_clauses, ext_mask, _ext_perm = a.make_image()
graph.add_clause([curr_lit], 0)
mask = [0]
graph.add_clause(all_goals, 1)
mask += [0] * len(all_goals)
for p in path:
    graph.add_clause([p], 2)
    mask += [0]    
graph.ini_var = 3
for axiom in ext_clauses:
    graph.reset_vars()
    graph.add_clause(axiom, 3) 
    print(graph.node_d.items())
mask += ext_mask