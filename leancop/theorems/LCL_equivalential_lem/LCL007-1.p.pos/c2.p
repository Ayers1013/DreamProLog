fof(c_0_6,conjecture,![A]:![B]:(is_a_theorem(A)|~is_a_theorem(B)|~is_a_theorem(equivalent(B,A)))).
cnf(condensed_detachment,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(ec_4,axiom,is_a_theorem(equivalent(equivalent(A,B),equivalent(B,A)))).
cnf(ec_5,axiom,is_a_theorem(equivalent(equivalent(equivalent(A,B),C),equivalent(A,equivalent(B,C))))).
cnf(c_0_4,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(c_0_5,axiom,is_a_theorem(equivalent(equivalent(A,B),equivalent(B,A)))).
cnf(c_0_8,axiom,is_a_theorem(equivalent(equivalent(equivalent(A,B),C),equivalent(A,equivalent(B,C))))).
