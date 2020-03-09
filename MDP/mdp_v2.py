'''
Utility file which defines the fundamental MDP data structures
that we will use to solve the different types of modelling exercises in this class.
This is an improved version which leverages the practices from this class and operates
on the MDP state tree as a single data structure, instead of the original approach
which was adapted from the method learned in CS221.
'''

import collections, random
import numpy as np
from typing import TypeVar, Mapping, Set, Tuple, Generic


'''
Custom type variables to be used in defining our data structures. These are
lifted from the in-class code. These are simply to make the code more understandable
i.e. so we can see how states map to actions from the type declarations.
'''
S = TypeVar('S')
A = TypeVar('A')



def retrieve_states(mdpData: Mapping[S, Any]) -> Set[S]:
	return set(mdpData.keys())

def retrieve_actions(mdpData: Mapping[S, Any]) -> Set[S]:


class MDPV2(Generic[S,A]):

	def __init__(self, 
		data: Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]],
		gamma: float) -> None:
		'''
		Constructor. Accepts the entire MDP network as a single data structure 
		where the top-level keys are states, the contents of each is another dict with
		actions as keys, and each of those contents is a tuple of the different successor
		states with probabilities. Also accepts the discount rate for the problem as a float. 
		'''

	    all_st = get_all_states(mdp_data)
	    check_actions = all(len(v) > 0 for _, v in mdp_data.items())
	    val_seq = [v2 for _, v1 in mdp_data.items() for _, (v2, _) in v1.items()]
	    return verify_transitions(all_st, val_seq) and check_actions


		self.gamma = gamma
		self.states = data.keys()
