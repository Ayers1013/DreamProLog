fof(c_0_23,conjecture,![A]:![B]:![C]:is_a_theorem(equivalent(equivalent(A,equivalent(B,equivalent(C,equivalent(A,B)))),C))).
cnf(condensed_detachment,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(yrm,axiom,is_a_theorem(equivalent(equivalent(A,B),equivalent(C,equivalent(equivalent(B,C),A))))).
cnf(c_0_3,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(c_0_4,axiom,is_a_theorem(equivalent(equivalent(A,B),equivalent(C,equivalent(equivalent(B,C),A))))).
