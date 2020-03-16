'''
File which defines the fundamental MDP data structures
that we will use to solve the different types of modelling exercises in this class.
This is an improved version which leverages the practices from this class and operates
on the MDP state tree as a single data structure, instead of the original approach
which was adapted from the method learned in CS221.
'''

import collections, random
import numpy as np
from typing import TypeVar, Mapping, Set, Tuple, Generic, Sequence, Callable, List
import mdpUtils as mu

'''
Custom type variables to be used in defining our data structures. These are
lifted from the in-class code. These are simply to make the code more understandable
i.e. so we can see how states map to actions from the type declarations.
'''
S = TypeVar('S')
A = TypeVar('A')
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


    def assign(self, state: S, action: A, prob: float) -> None:
        '''
        Helper method to assign a probability to a given state-action pair within the policy
        '''
        self.polData[state][action] = prob


    def assignDet(self, state: S, setAction: A) -> None:
        '''
        Helper method to deterministically assign an action to a given state
        (i.e. assigns a probability of 1 to that action and sets probability of all other actions to zero),
        helpful when using iterative methods
        '''
        for act in self.polData[state].keys():
            if act == setAction:
                self.polData[state][act] = 1.
            else:
                self.polData[state][act] = 0.


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
    Heavily adapted from CME241 class code for simplicity
    '''
    def __init__(self, mrpData: Mapping[S, Mapping[S, Tuple[float, float]]],
                gamma: float) -> None:
        '''
        Constructor to assign input data to relevant object data structures
        '''
        # TODO: Time permitting, refactor this so the data cleanly maps into the relevant
        # class member variables instead of lifting directly from the class code logic

        # Load the SSTff data and create the FULL rewards matrix R(s,s')
        # (From MRPRefined)
        a1, fullRewards, a3 = self.split_info(mrpData)   
        self.fullRewards = fullRewards 

        # Reformat the data for computing other properties
        # originally in MRP superclass from CME241 class code
        newData = {k: (v, a3[k]) for k, v in a1.items()}
        d1, d2 = mu.zip_dict_of_tuple(newData)

        self.states = mu.getStatesMRP(mrpData)
        self.transitions = mu.getTransitionsMRP(mrpData)
        self.gamma: float = gamma 
        self.rewards: Mapping[S, float] = d2
        self.terminal_states = self.get_terminal_states()
        self.nt_states_list = self.get_nt_states_list()
        self.trans_matrix = self.get_trans_matrix()
        self.rewards_vec: np.ndarray = self.get_rewards_vec()

    def split_info(self, info: SSTff) -> Tuple[SSf, SSf, Mapping[S, float]]:
        d = {k: mu.zip_dict_of_tuple(v) for k, v in info.items()}
        d1, d2 = mu.zip_dict_of_tuple(d)
        d3 = {k: sum(np.prod(x) for x in v.values()) for k, v in info.items()}
        return d1, d2, d3


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
        Taken from CME241 Class Code
        '''        
        sink = self.get_sink_states()
        return {s for s in sink if mu.isApproxEq(self.rewards[s], 0.0)}

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

    def get_rewards_vec(self) -> np.ndarray:
        """
        This rewards vec is only for the non-terminal states
        Taken from CME241 class code
        """
        return np.array([self.rewards[s] for s in self.nt_states_list])

    def get_value_func_vec(self) -> np.ndarray:
        """
        This value func vec is only for the non-terminal states
        Taken from CME241 class code
        """
        return np.linalg.inv(
            np.eye(len(self.nt_states_list)) - self.gamma * self.trans_matrix
        ).dot(self.rewards_vec)


class MDP_RL(Generic[S, A]):
    '''
    Interface for an MDP to be solved using a reinforcement learning
    (RL) algorithm. Since RL algorithms will learn information on-line from
    rollouts, we do not access rewards or probabilities directly. Instead 
    we present at each state actions possible from that state, and
    a dynamics model that allows us to "step" the state forward
    so that we can generate simulation episodes
    '''

    def __init__(self, stateActionDict: Mapping[S, Set[A]],
                    terminalStates: Set[S],
                    dynamics: Callable[[S, A], Tuple[S, float]]) -> None:
        '''
        Constructor
        '''
        # Dynamics accept a state-action pair and return a successor state and reward.
        # We exclude transition probability - the stochasticity of the dynamics
        # handles this for us. We also do not explicitly expose a "reward function",
        # i.e. calculation of the reward is confined to stepping forward in the dynamics
        self.dynamics = dynamics 

        # Dict telling us what actions are available from what states
        self.stateActionDict = stateActionDict 

        # Set of terminal states for the problem (need this to
        # test when to stop rollouts)
        self.terminalStates = terminalStates







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
        self.terminal_states = self.get_terminal_states()


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

        tr = mu.MDPDictToMRPDict(self.transitions, pol.polData)
        rew_ref = mu.MDPDictToMRPDict(
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


    def get_value_func_dict(self, pol: Policy)\
            -> Mapping[S, float]:
        '''
        Computes the state-value function V for each state
        For the MDP given a policy
        Taken from CME241 Class Code
        '''
        mrp_obj = self.getMRPFromPolicy(pol)
        value_func_vec = mrp_obj.get_value_func_vec()
        nt_vf = {mrp_obj.nt_states_list[i]: value_func_vec[i]
                 for i in range(len(mrp_obj.nt_states_list))}
        t_vf = {s: 0. for s in self.terminal_states}
        return {**nt_vf, **t_vf}


    def get_act_value_func_dict(self, pol: Policy)\
            -> Mapping[S, Mapping[A, float]]:
        '''
        Computes the state-action value function Q for each state-action pair
        For the MDP given a policy
        Adapted from CME241 Class Code, fixed to provide more robust implementation
        for T(s,a,s') and R(s,a,s') as per https://web.stanford.edu/class/archive/cs/cs221/cs221.1196/lectures/mdp2.pdf, page 5
        '''
        v_dict = self.get_value_func_dict(pol)

        q_dict = dict()

        for state, actDict in self.rewards.items():
            q_dict[state] = dict()
            for action, succDict in actDict.items():
                succSum = 0
                for succ, reward in succDict.items():
                    prob = self.transitions[state][action][succ]
                    succSum = succSum + prob*(reward + self.gamma*v_dict[succ])
                q_dict[state][action] = succSum

        return q_dict


if __name__ == "__main__":
    # MDP TEST
    # n = 10
    # # Test with lilypad problem from midterm
    # data = {
    #     i: {
    #         'A': {
    #             i - 1: (i / n, 0.),
    #             i + 1: (1. - i / n, 1. if i == n - 1 else 0.)
    #         },
    #         'B': {
    #             j: (1 / n, 1. if j == n else 0.)
    #             for j in range(n + 1) if j != i
    #         }
    #     } for i in range(1, n)
    # }
    # # Transition probabilities for edge cases at i=0, i=n
    # data[0] = {'A': {0: (1., 0.)}, 'B': {0: (1., 0.)}}
    # data[n] = {'A': {n: (1., 0.)}, 'B': {n: (1., 0.)}}

    # # Discount factor
    # gamma = 1.0
    # mdp = MDP(data, gamma)

    # print(mdp.states)
    # print(mdp.sa_dict)
    # print(mdp.transitions)
    # print(mdp.rewards)

    # MRP TEST
    data = {
        1: {1: (0.3, 9.2), 2: (0.6, 3.4), 3: (0.1, -0.3)},
        2: {1: (0.4, 0.0), 2: (0.2, 8.9), 3: (0.4, 3.5)},
        3: {3: (1.0, 0.0)}
    }
    mrp_refined_obj = MRP(data, 0.95)
    print(mrp_refined_obj.trans_matrix)
    print(mrp_refined_obj.rewards_vec)


