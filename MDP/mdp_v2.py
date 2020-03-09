'''
File which defines the fundamental MDP data structures
that we will use to solve the different types of modelling exercises in this class.
This is an improved version which leverages the practices from this class and operates
on the MDP state tree as a single data structure, instead of the original approach
which was adapted from the method learned in CS221.
'''

import collections, random
import numpy as np
from typing import TypeVar, Mapping, Set, Tuple, Generic
import mdp_utils as mu

'''
Custom type variables to be used in defining our data structures. These are
lifted from the in-class code. These are simply to make the code more understandable
i.e. so we can see how states map to actions from the type declarations.
'''
S = TypeVar('S')
A = TypeVar('A')


class Policy(Generic[S, A]):
    '''
    Generic data structure for a policy. The policy is a nested dict where
    the top-level dict maps states to various actions, and the lower-level dicts
    map actions to probabilities of taking that action (i.e. a stochastic policy).
    '''
    def __init__(self, 
                polData: Mapping[S, Mapping[A, float]]) -> None:
        '''
        Constructor. Verifies that the policy is well-formed 
        by checking that probabilities under each state sum to 1 before passing
        the data structure into the object, errors out otherwise
        '''
        if all(mu.isApproxEq(sum(action.values()), 1.0) for _, action in polData.items()):
            self.polData = polData
        else:
            raise ValueError("Policy improperly defined. Make sure action probabilities \
                for each state sum to 1")

    def getStateProbs(self, state: S) -> Mapping[A, float]:
        '''
        For a given state, returns a dict mapping actions 
        that can be taken from that state and the probability of the policy
        taking that action from that state 
        '''
        return self.polData[state]

    def getStateActionProb(self, state: S, action: A) -> float:
        '''
        Pulls the probability of the policy taking a particular state-action pair
        and returns zero if that action is never prescribed for that state.
        Using this function means a policy can only specify actions with 
        at each state 
        '''
        return self.getStateProbs(state).get(action, 0.0)


class MRP(Generic[S]):
    '''
    Generic data structure for an MRP. The MRP consists of states, successor states
    that can be reached from each state, and probabilities / rewards for each.
    Note that an MDP + Policy reduces to an MRP (this reductive mapping is implemented
    in the MDP class)
    '''
    def __init__(self, mrpData: Mapping[S, Mapping[S, Tuple[float, float]]],
                gamma: float) -> None:
        '''
        Constructor to assign input data to relevant object data structures
        '''
        self.states = mu.getStatesMRP()
        self.transitions = mu.getTransitionsMRP()
        self.rewards = mu.getRewardsMRP()


class MDPV2(Generic[S,A]):
    '''
    Generic data structure for an MDP. The MDP consists
    of states, actions to be taken at each state, and successor states.
    Each unique (s,a,s') triplet will have a unique probability of occurrence
    and reward. 
    '''

    def __init__(self, 
        data: Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]],
        gamma: float) -> None:
        '''
        Constructor. Accepts the entire MDP network as a single data structure 
        where the top-level keys are states, the contents of each is another dict with
        actions as keys, and each of those contents is a tuple of the different successor
        states with probabilities. Also accepts the discount rate for the problem as a float. 
        '''

        if not mu.verifyActions(data):
            raise ValueError("MDP actions not properly defined. Make sure that \
                all states have at least one action associated to them")

        self.gamma = gamma
        self.states = mu.getStates(data)
        self.actions = mu.getActions(data)
        self.transitions = mu.getTransitions(data)
        self.rewards = mu.getRewards(data)

    # def getMRPFromPolicy(self, pol: Policy) -> MRP:


if __name__ == "__main__":
    n = 10
    # Test with lilypad problem from midterm
    data = {
        i: {
            'A': {
                i - 1: (i / n, 0.),
                i + 1: (1. - i / n, 1. if i == n - 1 else 0.)
            },
            'B': {
                j: (1 / n, 1. if j == n else 0.)
                for j in range(n + 1) if j != i
            }
        } for i in range(1, n)
    }
    # Transition probabilities for edge cases at i=0, i=n
    data[0] = {'A': {0: (1., 0.)}, 'B': {0: (1., 0.)}}
    data[n] = {'A': {n: (1., 0.)}, 'B': {n: (1., 0.)}}

    # Discount factor
    gamma = 1.0
    mdp = MDPV2(data, gamma)

    print(mdp.states)
    print(mdp.actions)
    print(mdp.transitions)
    print(mdp.rewards)


