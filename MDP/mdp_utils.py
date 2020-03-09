'''
Utility file which contains helper functions that we will use both in our definitions
of MDP data structures as well sa in the algorithms we use to solve them.
'''
from typing import TypeVar, Mapping, Set, Tuple, Generic
S = TypeVar('S')
A = TypeVar('A')


def isApproxEq(a: float, b: float) -> bool:
	'''
	Test if two values are approximately equivalent within some margin
	of tolerance
	'''
	TOL = 1e-8
	return abs(a-b) <= TOL

def getStatesMRP(mrpData: Mapping[S, Mapping[S, Tuple[float, float]]]) -> Set[S]:
	'''
	Given an MRP dict, return all of the possible states as a set
	'''
	return set(mrpData.keys())

def getTransitionsMRP(mrpData: Mapping[S, Mapping[S, Tuple[float, float]]])\
						-> Mapping[S, Mapping[S, float]]:	
	'''
	Given an MRP dict, return a dict of non-negligibly probably transitions
	as another dict
	'''
	transProbs = dict()
	for state in mrpData.keys():
		transProbs[state] = dict()
		for succState in mrpData[state].keys():
			prob, _ = mrpData[state][succState]
			if not isApproxEq(prob, 0.0):
				transProbs[state][succState] = prob
	return transProbs 


def getRewardsMRP(mrpData: Mapping[S, Mapping[S, Tuple[float, float]]])\
						-> Mapping[S, float]:	
	'''
	Given an MRP dict, return a dict of rewards for each state-successor pair.
	'''
	rewards = dict()
	for state in mrpData.keys():
		rewards[state] = dict()
		for succState in mrpData[state].keys():
			reward, _ = mrpData[state][succState]
			if not isApproxEq(reward, 0.0):
				rewards[state][succState] = reward
	return rewards 


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
	transProbs = dict()
	for state in mdpData.keys():
		transProbs[state] = dict()
		for stateAction in mdpData[state].keys():
			transProbs[state][stateAction] = dict()
			for succState in mdpData[state][stateAction].keys():
				prob, _ = mdpData[state][stateAction][succState]
				if not isApproxEq(prob, 0.0):
					transProbs[state][stateAction][succState] = prob
	return transProbs

def getRewards(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) \
							-> Mapping[S, Mapping[A, Mapping[S, float]]]:
	'''
	Given an MDP dict, returns a dict which maps (state, action, successor)
	triples to that unique triple's reward. Note that rewards are usually
	unique to each state, but this is a more general representation
	for cases where the action drives the reward (e.g. in cases where actions
	have different "costs" framed as negative rewards)
	'''
	rewards = dict()
	for state in mdpData.keys():
		rewards[state] = dict()
		for stateAction in mdpData[state].keys():
			rewards[state][stateAction] = dict()
			for succState in mdpData[state][stateAction].keys():
				_, reward = mdpData[state][stateAction][succState]
				rewards[state][stateAction][succState] = reward
	return rewards


def verifyActions(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) -> bool:
	'''
	Verify that the MDP is properly formed in its actions. We will do so
	by verifying that every state has actions associated with it (note that this
	check is valid because we will handle sink states through the transition 
	matrix for such states only having probability mass in that same state)
	'''
	return all(len(v) > 0 for _, v in mdpData.items())



