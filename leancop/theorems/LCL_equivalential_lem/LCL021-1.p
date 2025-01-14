%--------------------------------------------------------------------------
% File     : LCL021-1 : TPTP v7.3.0. Released v1.0.0.
% Domain   : Logic Calculi (Equivalential)
% Problem  : XHK depends on XHN
% Version  : [McC92] axioms.
% English  : Show that the single Winker axiom XHK can be derived from the
%            single Winker axiom XHN.

% Refs     : [MW92]  McCune & Wos (1992), Experiments in Automated Deductio
%          : [McC92] McCune (1992), Email to G. Sutcliffe
%          : [Wos95] Wos (1995), Searching for Circles of Pure Proofs
% Source   : [McC92]
% Names    : EC-84 [MW92]

% Status   : Unsatisfiable
% Rating   : 0.17 v7.3.0, 0.00 v6.2.0, 0.33 v6.1.0, 0.64 v6.0.0, 0.44 v5.5.0, 0.69 v5.4.0, 0.72 v5.3.0, 0.80 v5.2.0, 0.62 v5.0.0, 0.60 v4.0.1, 0.43 v4.0.0, 0.29 v3.7.0, 0.43 v3.4.0, 0.40 v3.3.0, 0.00 v3.2.0, 0.33 v3.1.0, 0.50 v2.7.0, 0.62 v2.6.0, 0.57 v2.5.0, 0.71 v2.4.0, 0.86 v2.3.0, 0.57 v2.2.1, 0.89 v2.1.0, 1.00 v2.0.0
% Syntax   : Number of clauses     :    3 (   0 non-Horn;   2 unit;   2 RR)
%            Number of atoms       :    5 (   0 equality)
%            Maximal clause size   :    3 (   2 average)
%            Number of predicates  :    1 (   0 propositional; 1-1 arity)
%            Number of functors    :    4 (   3 constant; 0-2 arity)
%            Number of variables   :    5 (   0 singleton)
%            Maximal term depth    :    5 (   3 average)
% SPC      : CNF_UNS_RFO_NEQ_HRN

% Comments :
%--------------------------------------------------------------------------
cnf(condensed_detachment,axiom,
    ( ~ is_a_theorem(equivalent(X,Y))
    | ~ is_a_theorem(X)
    | is_a_theorem(Y) )).

%----Axiom by Winker
cnf(xhn,axiom,
    ( is_a_theorem(equivalent(X,equivalent(equivalent(Y,Z),equivalent(equivalent(Z,X),Y)))) )).

%----Axiom by Winker
cnf(prove_xhk,negated_conjecture,
    ( ~ is_a_theorem(equivalent(a,equivalent(equivalent(b,c),equivalent(equivalent(a,c),b)))) )).

%--------------------------------------------------------------------------
