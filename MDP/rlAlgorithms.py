import collections, random
from mdp import MDP, MDP_RL, Policy
import mdpUtils as mu
import numpy as np
from typing import TypeVar, Mapping, Set, Tuple, Generic, Sequence
S = TypeVar('S')
A = TypeVar('A')



class MDPAlgorithmRL:
    '''
    Generic algorithm class for solving an MDP.
    All algorithms we implement for solving MDPs will subclass this
    "interface" and contain a solve method to solve the MDP
    '''
    def __init__(self, mdp: MDP_RL):
        '''
        Interface function to accept an MDP to solve. The
        MDP must be specified in the appropriate interface for RL
        (i.e. without explicit reward / probability functions presented
        to this algorithmic agent)
        '''
        raise NotImplementedError("Override Me!")



class MonteCarloEG(MDPAlgorithmRL):
    '''
    Reinforcement learning solution using Monte-Carlo methods
    with Epsilon-Greedy policy improvement
    '''

    def __init__(self, rlMdp: MDP_RL, gamma: float) -> None:
        '''
        Constructor
        '''
        self.gamma = gamma # Discount rate for rewards
        self.mdp = rlMdp # MDP object (created using RL interface)


    def chooseRandomAction(self, actions: Sequence[A], eps: float = 0.1) -> A:
        '''
        Utility function for choosing a random action
        with probability (1-eps) + (eps/N_ACTIONS)
        '''
        # p = np.random.random()
        # if p < (1 - eps):
        #     return a
        # else:
        actions = list(actions)
        return actions[np.random.choice(len(actions))]


    def initializeAgent(self, startState: S, policy: Policy) -> None:
        '''
        Initialize the agent by setting a start state
        and declaring a policy
        '''
        self.state = startState
        self.policy = policy


    def isGameOver(self) -> bool:
        '''
        Helper method to check if we've finished the game, i.e.
        when we have reached a terminal state
        '''
        return (self.state in self.mdp.terminalStates)


    def episodeRollout(self, startState: S, policy: Policy) -> Tuple[Tuple[S, A, float]]:
        '''
        Performs a rollout of a single "episode" of the defined problem
        beginning from start state, take actions as prescribed by a policy
        until we reach a terminal state
        '''
        # Initialize the agent
        self.initializeAgent(startState, policy)
        action = self.chooseRandomAction(self.policy.polData[self.state].keys())

        # Initialize data structures for storing episode history
        succActionReward = []
        succActionReward.append((self.state, action, 0))


        # Forward Pass - Roll out the policy until the game ends
        # and collect info about rewards and state-action pairs
        while True:
            # Step forward. If we have a stochastic policy take a
            # weighted choice of actions from it
            self.state, reward = self.mdp.dynamics(self.state, action)

            # If game is over, assign zero reward and break
            if self.isGameOver():
                succActionReward.append((self.state, None, reward))
                break
            # Otherwise, append the rewards and choose a new action
            else:
                action = self.chooseRandomAction(self.policy.polData[self.state].keys())
                succActionReward.append((self.state, action, reward))


        # Backwards Pass
        G = 0
        firstEntry = True
        cumRewards = []

        for state, action, reward, in reversed(succActionReward):

            if firstEntry:
                firstEntry = False
            else:
                cumRewards.append((state, action, G))
            # Update reward
            G = reward + self.gamma*G

        cumRewards.reverse()

        return cumRewards


    def simulate_eg(self, startState: S, policy: Policy, T: int) -> Tuple[Policy, Mapping[S, float]]:

        Q = dict()
        returns = dict()

        for state in self.mdp.stateActionDict.keys():
            # For all non-terminal states, initialize the Q-function
            # and returns for every state-action pair associated with that state
            if state not in self.mdp.terminalStates:
                Q[state] = dict()
                for action in self.mdp.stateActionDict[state]:
                    Q[state][action] = 0
                    stateAction = (state, action)
                    returns[stateAction] = []
            # Pass if terminal state
            else:
                pass


        for t in range(T):
            print(t)
            # Roll out the policy for the given episode
            results = self.episodeRollout(startState, policy)

            seen_stateActions = set()

            for state, action, cumReward in results:
                # Pull the state-action pair from our episode history 
                stateAction = (state, action)

                # Check to see if this is not a state-action pair that we have seen previously
                if stateAction not in seen_stateActions:
                    origQ = Q[state][action]
                    returns[stateAction].append(cumReward)
                    # Overwrite the Q-value with the empirically observed reward from
                    Q[state][action] = np.mean(returns[stateAction])
                    # Add to the list of seen state-action pairs
                    seen_stateActions.add(stateAction)

            # Use the new information from the episode we have just run to perform
            # policy improvement by choosing the action for each state as the one that maximizes
            # the Q-value out of all possible actions from that state
            for state in policy.polData.keys():
                maxAction, maxValue = mu.maximizeOverDict(Q[state])
                policy.assignDet(state, maxAction)


        # Once we finish simulation runs we can also compute the optimal value function
        # after we have optimized the policy based on the empirical state-value 
        # function information we collected from running episodes
        optValue = dict()
        for state in policy.polData.keys():
            _, Vmax = mu.maximizeOverDict(Q[state])
            optValue[state] = Vmax
                
        return policy, optValue


