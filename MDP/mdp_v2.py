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


class MDPV2(Generic[S,A]):

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



