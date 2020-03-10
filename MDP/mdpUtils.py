'''
Utility file which contains helper functions that we will use both in our definitions
of MDP data structures as well sa in the algorithms we use to solve them.
'''
from typing import TypeVar, Mapping, Set, Tuple, Generic, Sequence, Callable, List, Any
S = TypeVar('S')
A = TypeVar('A')
X = TypeVar('X')


# Custom type variables from class code
VFType = Callable[[S], float]
QFType = Callable[[S], Callable[[A], float]]
PolicyType = Callable[[S], Callable[[int], Sequence[A]]]

VFDictType = Mapping[S, float]
QFDictType = Mapping[S, Mapping[A, float]]
PolicyActDictType = Callable[[S], Mapping[A, float]]

SSf = Mapping[S, Mapping[S, float]]
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]
STSff = Mapping[S, Tuple[Mapping[S, float], float]],
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]

FlattenedDict = List[Tuple[Tuple, float]]


######################################################
######################################################
'''
UTILITY FUNCTIONS AND TYPE VARIABLES FROM CME241 CLASS CODE
'''
def sum_dicts(dicts: Sequence[Mapping[X, float]]) -> Mapping[X, float]:
    return {k: sum(d.get(k, 0) for d in dicts)
            for k in set.union(*[set(d1) for d1 in dicts])}

def flatten_sasf_dict(sasf: SASf) -> FlattenedDict:
    return [((s, a, s1), f)
            for s, asf in sasf.items()
            for a, sf in asf.items()
            for s1, f in sf.items()]


def flatten_ssf_dict(ssf: SSf) -> FlattenedDict:
    return [((s, s1), f)
            for s, sf in ssf.items()
            for s1, f in sf.items()]


def unflatten_sasf_dict(q: FlattenedDict) -> SASf:
    dsasf = {}
    for (sas, f) in q:
        dasf = dsasf.get(sas[0], {})
        dsf = dasf.get(sas[1], {})
        dsf[sas[2]] = f
        dasf[sas[1]] = dsf
        dsasf[sas[0]] = dasf
    return dsasf


def unflatten_ssf_dict(q: FlattenedDict) -> SSf:
    dssf = {}
    for (ss, f) in q:
        dsf = dssf.get(ss[0], {})
        dsf[ss[1]] = f
        dssf[ss[0]] = dsf
    return dssf


def isApproxEq(a: float, b: float) -> bool:
	'''
	Test if two values are approximately equivalent within some margin
	of tolerance
	'''
	TOL = 1e-8
	return abs(a-b) <= TOL

def merge_dicts(d1: List[Tuple[Tuple, float]], 
				d2: List[Tuple[Tuple, float]], operation):
    merged = d1 + d2
    from itertools import groupby
    from operator import itemgetter
    from functools import reduce
    sortd = sorted(merged, key=itemgetter(0))
    grouped = groupby(sortd, key=itemgetter(0))
    return [(key, reduce(operation, [x for _, x in group])) for key, group in grouped]


######################################################
######################################################

def maximizeOverDict(data: Mapping[Any, float]):
  '''
  Utility function to maximize over a dict with
  '''
  max_key = None
  max_val = float('-inf')
  for k, v in data.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val



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


def MDPDictToMRPDict(mdpData: Mapping[S, Mapping[A, Mapping[S, float]]],
					polData: Mapping[S, Mapping[A, float]]) -> Mapping[S, Mapping[S, float]]:
	'''
	Accepts a set of either T(s,a,s') or R(s,a,s') values represented in a dict
	(i.e. either the transition probabilities or rewards) and a policy, and applies
	the policy to return the MRP-reduced version (i.e. to prescribe out the 
	action dependence so we have T(s,s') and R(s,s')).
	Adapted from CME241 class code 
	'''
	mrpData = dict()
	# Iterate through each state in the MDP
	for state, actDict in mdpData.items():
		sumDicts = []
		# Pull the possible actions in this state from the policy
		polActs = polData[state]
		for act, prob in polActs.items():
			succDict = actDict[act]
			worker = dict()
			for succState, reward in succDict:
				# Calculate the contribution to the value of the successor
				# state from that given state-action pair
				worker[succState] = reward*prob
			sumDicts.append(worker)
		# Sum the dicts we get from all possible actions prescribed by the policy
		# so we end up with a dict with the expected value of all the successor
		# states reachable from the given state
		mrpData[state] = sum_dicts(sumDicts)

	return mrpData

def getStates(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]])\
						-> Set[S]:
	'''
	Given an MDP dict, return all of the possible states defined by that MDP as a set.
	'''
	return set(mdpData.keys())


def getStateActDict(mdpData: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]) \
						-> Mapping[S, Set[A]]:
	'''
	Given an MDP dict, return a dict which simply maps all possible states
	to the actions that can be taken from each state, as a set
	'''
	return {state: set(action.keys()) for state, action in mdpData.items()}


def getAllActions(stateActDict: Mapping[S, Set[A]]) -> Set[S]:
	'''
	Given an MDP state-action dict calculated by getStateActDict, return all of the possible actions defined by that MDP as a set.
	'''
	actions = set()
	for state in stateActDict.keys():
		for stateAction in stateActDict[state]:
			actions.add(stateAction)
	return actions


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



