fof(c_0_30,conjecture,![A]:![B]:![C]:is_a_theorem(equivalent(A,equivalent(equivalent(B,equivalent(C,A)),equivalent(C,B))))).
cnf(condensed_detachment,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(pym,axiom,is_a_theorem(equivalent(equivalent(equivalent(A,equivalent(B,C)),B),equivalent(C,A)))).
cnf(c_0_3,axiom,(is_a_theorem(A)|~is_a_theorem(equivalent(B,A))|~is_a_theorem(B))).
cnf(c_0_4,axiom,is_a_theorem(equivalent(equivalent(equivalent(A,equivalent(B,C)),B),equivalent(C,A)))).
