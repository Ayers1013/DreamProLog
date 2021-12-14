:- use_module(library(assoc)).

:- ensure_loaded(leancop_step).

goals_length(State, L):-
    goals_list(State, Goals),
    length(Goals, L).

goals_size(State, S):-
    goals_list(State, Goals),
    maplist(term_size, Goals, Sizes),
    sumlist(Sizes, S).

text_actions(Features):-
    findall([ClauseText,Counter], (
                lit(_L2,L,C2,_G,_Key,_Vars,Counter),
                Clause = [L|C2],
                numbervars(Clause, 1000, _),
                text_from_path(Clause, "", ClauseText)
            ), Features
           ).

text_actions_mask(State, Features, Mask):-
    text_actions(Features0),
    State=state(_Tableau, Actions, _Result),

    findall(N, (
                member([_,Counter], Features0),
                ( nth0(N,Actions,ext(_,_,_,_,_,Counter)) -> true
                ; N = -1
                )
            ), Mask
           ),

    findall(F, member([F,_], Features0), Features).
    
text_features(StateOrig, Features):-
    copy_term(StateOrig, State),

    State=state(Tableau, _Actions, _Result),
    tab_comp(goal, Tableau, Goal),
    tab_comp(path, Tableau, Path),
    tab_comp(todos, Tableau, Todos),
    numbervars([Goal, Path, Todos], 1000, _),

    ( Goal = [success] -> Features = []
    ; Goal = [failure] ->
      findall("FAILURE", between(1,99, _), Features)
    ; Todos2 = [[Goal, Path, _] |Todos],
      text_from_todo_list(Todos2, Features)
    ).

text_from_todo_list([], []).
text_from_todo_list([[Cla, Path, _]|Todos], Features):-
    text_from_clause(Cla, Path, Features0),
    append(Features0, Features1, Features),
    text_from_todo_list(Todos, Features1).

text_from_clause([], _, []).
text_from_clause([Goal|Cla], Path, [F|Features]):-    
    text_from_path([Goal|Path],"", F),
    text_from_clause(Cla, Path, Features).

text_from_path([], Text, Text).
text_from_path([P|Ps], Acc, Text):-
    text_from_lit(P, Acc, Acc1),
    write_after(Acc1, "EOP", Acc2),
    text_from_path(Ps, Acc2, Text).

text_from_lit(Lit, Acc, Text):-
    ( Lit = -NegLit ->
      write_after(Acc, "NEG", Acc1),
      text_from_lit(NegLit, Acc1, Text)
    ; atomic(Lit) ->
      write_after(Acc, Lit, Text)
    ; Lit = '$VAR'(N) ->
      atom_concat("var_", N, Var),
      write_after(Acc, Var, Text)
    ; Lit = I^Args ->
      atom_concat("skolem_", I, Skolem),
      write_after(Acc, Skolem, Acc1),
      length(Args, L),
      write_after(Acc1, L, Acc2),
      text_from_lit_list(Args, Acc2, Text)
    ; Lit =.. [H|Args],
      write_after(Acc, H, Acc1),
      length(Args, L),
      write_after(Acc1, L, Acc2),
      text_from_lit_list(Args, Acc2, Text)
    ).

text_from_lit_list([], Acc, Acc).
text_from_lit_list([L|Ls], Acc, Text):-
    text_from_lit(L, Acc, Acc1),
    text_from_lit_list(Ls, Acc1, Text).

write_after(A, B, C):-
    atom_concat(A, " ", A1),
    atom_concat(A1, B, C).

    


simple_features(StateOrig, Features):-
    copy_term(StateOrig, State),

    State=state(Tableau, _Actions, _Result),
    tab_comp(goal, Tableau, Goal),
    tab_comp(path, Tableau, Path),
    tab_comp(todos, Tableau, Todos),
    numbervars([Goal, Path, Todos], 1000, _),

    goals_list(Todos, [Goal], AllGoals),
    length(Path, PathLen),
    length(AllGoals, NumGoals),

    goalStats(AllGoals, GoalsSymbolSize, MaxGoalSize, MaxGoalDepth, _TopSymbol1,  _TopSymbol2, TopFrequency1, TopFrequency2),
    varStats(AllGoals, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc),
    action_count_total(State, ACT),

    Features = [
        NumGoals,
		GoalsSymbolSize,
		MaxGoalSize,
		MaxGoalDepth,
		PathLen,
		TopFrequency1,
		TopFrequency2,
        ACT,
        NumVar,
        NumVarOcc,
        NumVarOneOcc,
        NumVarMoreOcc,
        MostOcc,
        LeastOcc
    ].


%% EmbType in [both, state_only]
logic_embed(StateOrig,FHash,EmbType, EmbStateP,EmbStateV,EmbActions):-
    copy_term(StateOrig, State),

    State=state(Tableau, Actions, _Result),
    mc_param(n_dim,FDim),

    tab_comp(goal, Tableau, Goal),
    tab_comp(path, Tableau, Path),
    tab_comp(todos, Tableau, Todos),
    tab_comp(lem, Tableau, Lem),

    ( mc_param(collapse_vars,1) -> collapse_vars([Goal, Path, Lem, Todos, Actions])
    ; numbervars([Goal, Path, Lem, Todos], 1000, _VarCount),
      numbervars_list(Actions, 1000)
    ),


    cached_embed(Goal,FHash,FDim,0,EGoal),
    cached_embed(Path,FHash,FDim,FDim,EPath),    
    goals_list(Todos, [Goal], AllGoals),
    cached_embed(AllGoals,FHash,FDim,0,EAllGoals),

    
    
    
    Offset1 is 2*FDim,
    PFeatures1 = [EGoal, EPath],
    VFeatures1 = [EAllGoals, EPath],

    
    ( mc_param(lemma_features,1) ->
      cached_embed(Lem, FHash, FDim, Offset1, ELem),
      Offset_todos is Offset1 + FDim,
      cached_embed(Todos, FHash, FDim, Offset_todos, ETodos),
      Offset2 is Offset_todos + FDim,
      append(PFeatures1, [ELem, ETodos], PFeatures2),
      append(VFeatures1, [ELem, ETodos], VFeatures2)
    ; Offset2 = Offset1,
      PFeatures2 = PFeatures1,
      VFeatures2 = VFeatures1
    ),

    ( mc_param(subst_features,1) ->
      tab_comp(subst, Tableau, Subst),
      subst2preds(Subst, SubstPreds),
      cached_embed(substitution(SubstPreds), FHash, FDim, Offset2, ESubstPreds),
      Offset3 is Offset2 + FDim,
      append(PFeatures2, [ESubstPreds], PFeatures3),
      append(VFeatures2, [ESubstPreds], VFeatures3)
    ; Offset3 = Offset2,
      PFeatures3 = PFeatures2,
      VFeatures3 = VFeatures2
    ),

    merge_features_list(PFeatures3, [], EmbStateP0),
    merge_features_list(VFeatures3, [], EmbStateV0),
    
    length(Path, PathLen),
    length(AllGoals, NumGoals),
    goalStats(AllGoals, GoalsSymbolSize, MaxGoalSize, MaxGoalDepth, TopSymbol1,  TopSymbol2, TopFrequency1, TopFrequency2),

    varStats(AllGoals, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc),

    I1 = Offset3,
    I2 is I1 + 1,
    I3 is I1 + 2,
    I4 is I1 + 3,
    I5 is I1 + 4,
    I6 is I1 + 5,
    I7 is I1 + 6,
    I8 is I1 + 7,
    I9 is I1 + 8,
    I10 is I1 + 9,
    I11 is I1 + 10,
    I12 is I1 + 11,
    I13 is I1 + 12,
    I14 is I1 + 13,
    I15 is I1 + 14,
    I16 is I1 + 15,
    action_count_total(State, ACT),
    
    GlobalFeatures = [
        [I1, NumGoals],
		[I2, GoalsSymbolSize],
		[I3, MaxGoalSize],
		[I4, MaxGoalDepth],
		[I5, PathLen],
		[I6, TopSymbol1],
		[I7, TopSymbol2],
		[I8, TopFrequency1],
		[I9, TopFrequency2],
        [I10, ACT],
        [I11, NumVar],
        [I12, NumVarOcc],
        [I13, NumVarOneOcc],
        [I14, NumVarMoreOcc],
        [I15, MostOcc],
        [I16, LeastOcc]
    ],
    append(EmbStateP0, GlobalFeatures, EmbStateP),
    append(EmbStateV0, GlobalFeatures, EmbStateV),

    ( EmbType = both ->
      Offset = I16,
      cached_embed_list(Actions, FHash, FDim, Offset, EmbActions)
    %% length(Actions, ALen), 
    %% logic_embed_successors(0, ALen, State, FHash, EmbActions)
    ; true
    ).

logic_embed_successors(I,I,_,_,[]):- !.
logic_embed_successors(I, ALen, State, FHash, [E|EmbActions]):-
    I < ALen,
    copy_term(State, State2),
    logic_step(State2,I,ChildState),
    logic_embed(ChildState,FHash,state_only,_,E,_),
    I1 is I+1,
    logic_embed_successors(I1, ALen, State, FHash, EmbActions).
    



%% number of variable instantiations is not available for us
goalStats(Goals, GoalsSymbolSize, MaxGoalSize, MaxGoalDepth, TopSymbol1,  TopSymbol2, TopFrequency1, TopFrequency2):-
    maplist(term_size, Goals, GoalSizes),
    maplist(term_depth, Goals, GoalDepths),
    sumlist(GoalSizes, GoalsSymbolSize),
    max_list(GoalSizes, MaxGoalSize),
    max_list(GoalDepths, MaxGoalDepth),
    top_two_symbols(Goals, _, _, TopFrequency1, TopFrequency2, TopSymbol1, TopSymbol2).

% goals is ground (numbervars has been called on it)
varStats(Goals, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc):-
    ground(Goals),
    count_var_occ(Goals, [], VarOccList),
    countVars(VarOccList, 0, 0, 0, 0, 0, 10000, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc).

count_var_occ(X, Acc, Acc):-
    atomic(X), !.
count_var_occ([X|Xs], Acc, Result):-
    !,
    count_var_occ(X, Acc, Acc2),
    count_var_occ(Xs, Acc2, Result).
count_var_occ('$VAR'(N), Acc, Result):-
    !,
    add_var(Acc, N, Result).
count_var_occ(X, Acc, Result):-
    X =.. [_|Args],
    count_var_occ(Args, Acc, Result).

add_var([], N, [N-1]).
add_var([N-C|Rest], N, [N-C2|Rest]):-
    !,
    C2 is C + 1.
add_var([X|Rest], N, [X|Rest2]):-
    add_var(Rest, N, Rest2).

countVars([], NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc).
countVars([_V-C|Rest], Acc1, Acc2, Acc3, Acc4, Acc5, Acc6, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc):-
    Acc11 is Acc1 + 1,
    Acc21 is Acc2 + C,
    ( C = 1 -> Acc31 is Acc3 + 1, Acc41 = Acc4
    ; Acc31 = Acc3, Acc41 is Acc4 + 1
    ),
    Acc51 is max(Acc5, C),
    Acc61 is min(Acc6, C),
    countVars(Rest, Acc11, Acc21, Acc31, Acc41, Acc51, Acc61, NumVar, NumVarOcc, NumVarOneOcc, NumVarMoreOcc, MostOcc, LeastOcc).


collapse_vars(X):-
    ( var(X) -> X=var
    ; atomic(X) -> true
    ; X = [_|_] -> maplist(collapse_vars, X)
    ; X =.. [_|Args], maplist(collapse_vars, Args)
    ).

goals_list(state(Tableau, _, _), Goals):-
    tab_comp(goal, Tableau, Goal),
    tab_comp(todos, Tableau, Todos),
    goals_list(Todos, [Goal], GoalsList),
    append(GoalsList, Goals).
goals_list([], Goals, Goals).
goals_list([T|Ts], Acc, Goals):-
    T = [Goal,_Path,_Lem],
    goals_list(Ts, [Goal|Acc], Goals).


term_size(Term, Size):-
    ( var(Term) -> Size = 1
    ; Term =.. ['$VAR'|_] -> Size = 1
    ; atomic(Term) -> Size = 1
    ; Term =.. [_|Tail],
      term_size_list(Tail, 1, Size)
    ).
term_size_list([], Acc, Acc).
term_size_list([T|Ts], Acc, Size):-
    term_size(T,Size1),
    Acc1 is Size1 + Acc,
    term_size_list(Ts, Acc1, Size).

term_depth(Term, 0):-
    \+ compound(Term), !.
term_depth(Term, 0):-
    Term =.. ['$VAR'|_], !.
term_depth(Term, Depth):-
    compound(Term),
    Term =.. [_|Tail],
    maplist(term_depth, Tail, Depths),
    max_list(Depths, Max),
    Depth is Max + 1.

action_count_total(state(Tableau, Actions, _), AC):-
    length(Actions, AC0),
    tab_comp(todos, Tableau, Todos),
    action_count_total(Todos, AC0, AC).
action_count_total([], AC, AC).
action_count_total([T|Ts], Acc, AC):-
    T = [Goal, Path, _Lem],
    valid_actions_filter(Goal, Path, Actions),
    length(Actions, AC0),
    Acc1 is Acc + AC0,
    action_count_total(Ts, Acc1, AC).

numbervars_list([],_).
numbervars_list([A|As], From):-
    numbervars(A, From, _),
    numbervars_list(As, From).


state2gnnInput(State, GnnInput):-
    State=state(Tableau, Actions, Result),
    tab_comp(goal, Tableau, Goal),
    tab_comp(path, Tableau, Path),
    tab_comp(todos, Tableau, Todos),
    goals_list(Todos, [Goal], AllGoals), append(AllGoals, AllGoals1),
    ( Result= -1 -> AllGoals2 = []
    ; Result=1 -> append(_,[H],Path), AllGoals2 = [H]
    ; AllGoals2 = AllGoals1
    ),
    goal2gnnInput(Actions, Goal, Path, AllGoals2, GnnInput).


goal2gnnInput(Actions, Goal, Path, AllGoals, GnnInput):-
    Goal = [CurrLit|_],
    actions2Clauses(Actions, ExtClauses, ExtMask, ExtPerm),
    once(term2list(CurrLit, CurrLit1)),
    once(term2list(AllGoals,AllGoals1)),
    once(term2list(Path,Path1)),
    GnnInput = [CurrLit1, Path1, AllGoals1, ExtClauses, ExtMask, ExtPerm].

% todo support for other action types (red, para)
actions2Clauses(Actions, Clauses, Mask, Permutation):-
    all_clauses(Clauses), !,
    append(Clauses, AllClauseLiterals),
    findall(M, (
                nth0(N, AllClauseLiterals, _),
                ( member(ext(_, _, _, _, _, N), Actions) -> M = 1 ; M=0 )
            ), Mask
           ),
    findall(P, (
                member(ext(_, _, _, _, _, Counter), Actions),
                length(Prefix, Counter),
                append(Prefix, _, Mask),
                sum_list(Prefix, P)
            ), Permutation
           ).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                                                                                                                                     

top_two_symbols(TermList, TopSymbol1, TopSymbol2, TopFrequency1, TopFrequency2, TopHash1, TopHash2):-
    symbol_frequencies(TermList, Assoc),
    assoc_to_list(Assoc, Pairs),
    find_greatest_values(Pairs, xxx, xxx, 0, 0, TopSymbol1, TopSymbol2, TopFrequency1, TopFrequency2),
    term_hash(TopSymbol1, TopHash1),
    term_hash(TopSymbol2, TopHash2).

find_greatest_values([], S1, S2, F1, F2, S1, S2, F1, F2).
find_greatest_values([S-F|Pairs], AccS1, AccS2, AccF1, AccF2, S1, S2, F1, F2):-
    ( F > AccF1 -> find_greatest_values(Pairs, S, AccS1, F, AccF1, S1, S2, F1, F2)
    ; F > AccF2 -> find_greatest_values(Pairs, AccS1, S, AccF1, F, S1, S2, F1, F2)
    ; find_greatest_values(Pairs, AccS1, AccS2, AccF1, AccF2, S1, S2, F1, F2)
    ).

symbol_frequencies(TermList, Assoc):-
    empty_assoc(Assoc0),
    symbol_frequencies2_list(TermList, Assoc0, Assoc).

symbol_frequencies2(Term, Assoc0, Assoc):-
    ( var(Term) -> Assoc = Assoc0
    ; Term =.. ['$VAR'|_] -> Assoc = Assoc0
    ; atomic(Term) -> bag_insert(Term, Assoc0, Assoc)
    ; Term =.. [H|T],
      bag_insert(H, Assoc0, Assoc1),
      symbol_frequencies2_list(T, Assoc1, Assoc)
    ).

symbol_frequencies2_list([], Assoc, Assoc).
symbol_frequencies2_list([L|Ls], Assoc0, Assoc):-
    symbol_frequencies2(L, Assoc0, Assoc1),
    symbol_frequencies2_list(Ls, Assoc1, Assoc).

bag_insert(X,In,Out):-
    (get_assoc(X, In, V) ->
     V1 is V + 1
    ;
     V1 is 1
     ),
    put_assoc(X, In, V1, Out).
