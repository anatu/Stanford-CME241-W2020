'''
Utility file which contains helper functions that we will use both in our definitions
of MDP data structures as well sa in the algorithms we use to solve them.
'''
from typing import TypeVar, Mapping, Set, Tuple, Generic
S = TypeVar('S')
A = TypeVar('A')


def getStates(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]])\
						-> Set[S]:
	'''
	Given an MDP dict, return all of the possible states defined by that MDP as a set.
	'''
	return set(mdpData.keys())

def getActions(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) \
						-> Set[S]:
	'''
	Given an MDP dict, return all of the possible actions defined by that MDP as a set.
	'''
	actions = set()
	for state in mdpData.keys():
		for stateAction in mdpData[state].keys():
			actions.add(stateAction)
	return actions

def getStateActDict(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) \
						-> Mapping[S, Set[A]]:
	'''
	Given an MDP dict, return a dict which simply maps all possible states
	to the actions that can be taken from each state, as a set
	'''
	return {state: set(action.keys()) for state, action in mdpData.items()}

def getTransitions(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) \
							-> Mapping[S, Mapping[A, Mapping[S, float]]]:
	'''
	Given an MDP dict, returns a dict which maps (state, action, successor)
	triples to that unique triple's transition probability. This "leans out"
	the transition probability dict by not storing transition probabilities 
	which are at or near zero (based on some tolerance).
	'''
	TOL = 1e-8
	transProbs = dict()
	for state in mdpData.keys():
		transProbs[state] = dict()
		for stateAction in mdpData[state].keys():
			transProbs[state][stateAction] = dict()
			for succState in mdpData[state][stateAction].keys():
				prob, _ = mdpData[state][stateAction][succState]
				if not (abs(prob-0.0) <= TOL):
					transProbs[state][stateAction][succState] = prob
	return transProbs


def verifyActions(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) -> bool:
	'''
	Verify that the MDP is properly formed in its actions. We will do so
	by verifying that every state has actions associated with it (note that this
	check is valid because we will handle sink states through the transition 
	matrix for such states only having probability mass in that same state)
	'''
	return all(len(v) > 0 for _, v in mdpData.items())



