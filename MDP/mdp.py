'''
File which defines the fundamental MDP data structures
that we will use to solve the different types of modelling exercises in this class.
This is an improved version which leverages the practices from this class and operates
on the MDP state tree as a single data structure, instead of the original approach
which was adapted from the method learned in CS221.
'''

import collections, random
import numpy as np
from typing import TypeVar, Mapping, Set, Tuple, Generic, Sequence
import mdpUtils as mu

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
    if the policy is deterministic (e.g. for an optimal policy), then each state key will only
    have a single action beneath it with a probability of 1.
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
        Using this function means we can write deterministic policies without having to define
        a bunch of zeros for all the other states 
        '''
        return self.getStateProbs(state).get(action, 0.0)

    def __repr__(self):
        '''
        Helper function to print out policy data when print is called instead of an object reference
        '''
        return self.polData.__repr__()

    def __str__(self):
        '''
        Helper function to print out policy data when print is called instead of an object reference
        '''
        return self.polData.__str__()


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
        self.gamma = gamma 
        self.states = mu.getStatesMRP(mrpData)
        self.transitions = mu.getTransitionsMRP(mrpData)
        self.rewards = mu.getRewardsMRP(mrpData)
        self.terminal_states = self.get_terminal_states()
        self.nt_states_list = self.get_nt_states_list()
        self.trans_matrix = self.get_trans_matrix()


    def get_sink_states(self) -> Set[S]:
        '''
        Retrieves all sink states for the MRP
        Taken from CME241 class code
        '''
        return {k for k, v in self.transitions.items()
                if len(v) == 1 and k in v.keys()}


    def get_nt_states_list(self) -> Sequence[S]:
        '''
        Retrieves all non-terminal states for the MRP
        Taken from CME241 class code
        '''
        return [s for s in self.states
                if s not in self.terminal_states]

    def get_terminal_states(self) -> Set[S]:
        '''
        Helper method to calculate all terminal states for a given MDP as a set.
        Terminal sates are sink states, but they have a reward of zero
        Adapted from CME241 Class Code
        '''
        sink = self.get_sink_states()
        result = set()
        for s in sink:
            print(self.rewards[s])
            _, rMax = mu.maximizeOverDict(self.rewards[s])
            if mu.isApproxEq(0.0, rMax):
               result.add(s)
        return result


    def get_trans_matrix(self) -> np.ndarray:
        """
        This transition matrix is only for the non-terminal states
        Taken from CME241 class code
        """
        n = len(self.nt_states_list)
        m = np.zeros((n, n))
        for i in range(n):
            for s, d in self.transitions[self.nt_states_list[i]].items():
                if s in self.nt_states_list:
                    m[i, self.nt_states_list.index(s)] = d
        return m

    # TODO: Implement Linear Solution for the Value Function np.matmul(np.linalg.inv(I-gamma*P),R)   
    # where P is the transition prob matrix P(s,s') and R is the rewards matrix R(s,s')



class MDP(Generic[S,A]):
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


        # Store individual lists of relevant information
        self.gamma = gamma
        self.states = mu.getStates(data)
        self.sa_dict = mu.getStateActDict(data)
        self.actions = mu.getAllActions(self.sa_dict)
        self.transitions = mu.getTransitions(data)
        self.rewards = mu.getRewards(data)

    # TODO: Fix this, there appears to be an issue with how the rewards transfer over
    # Potentially just implement from scratch using https://towardsdatascience.com/reinforcement-learning-demystified-markov-decision-processes-part-1-bf00dda41690
    def getMRPFromPolicy(self, pol: Policy) -> MRP:
        '''
        Uses the provided policy to reduce the given MDP to an MRP by assigning 
        out the actions in the problem based on what is prescribed by the policy.
        Taken directly from CME241 class code
        '''
        flat_transitions = mu.flatten_sasf_dict(self.transitions)
        flat_rewards_refined = mu.flatten_sasf_dict(self.rewards)

        flat_exp_rewards = mu.merge_dicts(flat_rewards_refined, flat_transitions, lambda x, y: x * y)
        exp_rewards = mu.unflatten_sasf_dict(flat_exp_rewards)

        tr = mu.mdp_rep_to_mrp_rep1(self.transitions, pol.polData)
        rew_ref = mu.mdp_rep_to_mrp_rep1(
            exp_rewards,
            pol.polData
        )
        flat_tr = mu.flatten_ssf_dict(tr)
        flat_rew_ref = mu.flatten_ssf_dict(rew_ref)
        flat_norm_rewards = mu.merge_dicts(flat_rew_ref, flat_tr, lambda x, y: x / y)
        norm_rewards = mu.unflatten_ssf_dict(flat_norm_rewards)

        return MRP(
            {s: {s1: (v1, norm_rewards[s][s1]) for s1, v1 in v.items()}
             for s, v in tr.items()},
            self.gamma
        )



    def get_sink_states(self) -> Set[S]:
        '''
        Helper method to get all sink states for the given MDP as a set.
        We consider sink states ones which have only one possible transition, which is back to itself.
        From CME241 Class Code
        '''
        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }


    def get_terminal_states(self) -> Set[S]:
        '''
        Helper method to calculate all terminal states for a given MDP as a set.
        Terminal sates are sink states, but they have a reward of zero
        Adapted from CME241 Class Code
        '''
        sink = self.get_sink_states()
        result = set()
        for s in sink:
            for act in self.rewards[s].keys():
                _, rMax = mu.maximizeOverDict(self.rewards[s][act])
            if mu.isApproxEq(0.0, rMax):
                result.add(s)
        return result
        # return {s for s in sink if mu.isApproxEq(r, 0.0) for _, r in self.rewards[s].items())}


    # TODO: Implement helper methods to calculate V and Q based on the in-class code.


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
    mdp = MDP(data, gamma)

    print(mdp.states)
    print(mdp.sa_dict)
    print(mdp.transitions)
    print(mdp.rewards)


