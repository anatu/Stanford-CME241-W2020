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
		p = np.random.random()
		if p < (1 - eps):
		return a
		else:
		return np.random.choice(actions)


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
        return (state in self.mdp.terminalStates)


	def episodeRollout(self, startState: S, policy: Policy) -> Tuple[Tuple[S, A, float]]:
		'''
		Performs a rollout of a single "episode" of the defined problem
		beginning from start state, take actions as prescribed by a policy
		until we reach a terminal state
		'''
		# Initialize the agent
		self.initalizeAgent(startState, policy)
		action = self.chooseRandomAction(self.policy[state].keys())

		# Initialize data structures for storing episode history
		succActionReward = []
		succActionReward.append((self.state, action, 0))


		# Forward Pass - Roll out the policy until the game ends
		# and collect info about rewards and state-action pairs
		while True:
			# Step forward. If we have a stochastic policy take a
			# weighted choice of actions from it
			self.state, reward = mdp.dynamics(self.state, action)

			# If game is over, assign zero reward and break
			if self.isGameOver():
				succActionReward.append((state, None, reward))
				break
			# Otherwise, append the rewards and choose a new action
			else:
				action = chooseRandomAction(self.policy[state].keys())
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





