import collections, random
from mdp import MDP, Policy
import mdpUtils as mu
import numpy as np
from typing import TypeVar, Mapping, Set, Tuple, Generic, Sequence
S = TypeVar('S')
A = TypeVar('A')



class MDPAlgorithm:
	'''
	Generic algorithm class for solving an MDP.
	All algorithms we implement for solving MDPs will subclass this
	"interface" and contain a solve method to solve the MDP
	'''
	def solve(self, mdp: MDP):
		'''
		Interface function to solve a given MDP passed to the method.
		This should explicitly set as a class member variable both the optimal policy, as well as the
		optimal value function resulting from that policy
		'''
		raise NotImplementedError("Override Me!")


class ValueIteration(MDPAlgorithm):
	'''
	Algorithm to solve a given MDP using Value Iteration. Broadly this consists of two steps.
	First, the value function vector is initialized, and we iteratively assign into each state the maximum
	value function resulting from any of the actions we could take in that state (based on expected reward, recursively
	calculated), and repeat until convergence of the value function vector. 

	'''
	def __init__(self, tol: float):
		# Set the convergence tolerance for the algorithm
		self.tol = tol


	def solve(self, mdp: MDP) -> None:

		# Initialize the value function and policy data structures
		self.V = dict()
		polData = dict()
		for state in mdp.states:
			self.V[state] = 0.5
			polData[state] = dict()

		# Set the terminal-state value function entries to zero
		termStates = mdp.get_terminal_states()
		for ts in termStates:
			self.V[ts] = 0

		# Inflate the convergence epsilon to start so we can enter the while loop
		eps = self.tol*1e4
		
		# Enter into convergence loop to converge the value function according to the Bellman equation
		while eps >= self.tol:
			# Store the value function from the previous iteration
			# so we can recalculate epsilon
			lastV = self.V

			# Iterate through each of the states and store the maximum value function
			for state in mdp.states:
				V_cands = dict()
				# Iterate through each of the actions we can take from this state
				for action in mdp.sa_dict[state]:
					Vi = 0
					# Iterate through each possible successor state and calculate expected reward across each
					for succState in mdp.rewards[state][action].keys():
						P_sas = mdp.transitions[state][action][succState]
						R_sas = mdp.rewards[state][action][succState]						
						Vi = Vi + P_sas*(R_sas + self.V[succState])
					# Append the value-function for that action to the list
					V_cands[action] = Vi

				aOpt, vOpt = mu.maximizeOverDict(V_cands)
				# Assign the value function for that state as the maximum across all action candidates
				self.V[state] = vOpt
				polData[state][aOpt] = 1.0

			# Recalculate convergence criterion
			_, old = mu.maximizeOverDict(lastV)
			_, new = mu.maximizeOverDict(self.V)
			eps = abs(new - old)


		# After convergence, create the final policy data structure
		self.pi = Policy(polData)

		return None




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

    vi = ValueIteration(tol = 1e-8)
    vi.solve(mdp)
    print(vi.V)
    print(vi.pi)
    # print(mdp.calculateQ(vi.pi))





