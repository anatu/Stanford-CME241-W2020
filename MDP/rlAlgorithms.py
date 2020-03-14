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


	def chooseRandomAction(self, eps=0.1) -> A:
		'''
		Utility function for choosing a random action
		with probability (1-eps) + (eps/N_ACTIONS)
		'''
		p = np.random.random()
		if p < (1 - eps):
		return a
		else:
		return np.random.choice(actions)


	def initializeAgent(self, startState: S, policy: Mapping[S, Mapping[A, float]]) -> None:
		'''
		Initialize the agent by setting a start state
		and declaring a policy
		'''
		self.state = startState
		self.policy = policy


	def episodeRollout(self) -> Tuple[Tuple[S, A, float]]:
		'''
		Performs 
		'''