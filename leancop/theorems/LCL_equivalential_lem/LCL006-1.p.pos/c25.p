fof(c_0_32,conjecture,![A]:![B]:![C]:is_a_theorem(equivalent(A,equivalent(equivalent(equivalent(B,equivalent(A,C)),C),B)))).
cnf(condensed_detachment,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(ec_4,axiom,is_a_theorem(equivalent(equivalent(A,B),equivalent(B,A)))).
cnf(ec_5,axiom,is_a_theorem(equivalent(equivalent(equivalent(A,B),C),equivalent(A,equivalent(B,C))))).
cnf(c_0_4,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(c_0_5,axiom,is_a_theorem(equivalent(equivalent(A,B),equivalent(B,A)))).
cnf(c_0_6,axiom,is_a_theorem(equivalent(equivalent(equivalent(A,B),C),equivalent(A,equivalent(B,C))))).
