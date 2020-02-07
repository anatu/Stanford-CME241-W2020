import os
import re
import sys
sys.path.append('../')
from MDP import util




class MertonMDP():

	# Declare the parameters for the problem
	# Initial wealth (W0)
	# Relative risk-aversion (gamma)
	# Distribution parameters for the risky asset (mu, sigma)
	def __init__(self, W0, gamma, mu, sigma):
		self.W0 = W0
		self.gamma = gamma
		self.mu = mu
		self.sigma = sigma

		# Utility discount rate
		self.rho = 0.9
		# Number of time steps
		self.T = 1000

    # Return the start state.
    # States are tuples of (W,t)
    # so our start state is initial wealth at zero time
    def startState(self): 
    	return (self.W0, 0)

    # Utility function for the problem
    def UtilityOfConsumption(c):
    	if self.gamma = 1:
    		return log(c)
    	else:
    		return (c^(1-gamma))/(1-gamma) 

    # Return boolean check for terminal state
    def isTerminalState(state):
    	return state[1] == T

    # Actions define are the amount of wealth consumed at a given time (c)
    # as well as the fractional allocation between risky and riskless assets (pi)
    # where pi goes to risky asset and (1-pi) goes to riskless asset
    # Use the decisions for c, pi at each timestep to calculate resultant rewards



    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)
        # print self.states




