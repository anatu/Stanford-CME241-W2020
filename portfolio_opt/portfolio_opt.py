import os
import re
import sys
sys.path.append('../')
from MDP import util


# See https://en.wikipedia.org/wiki/Merton%27s_portfolio_problem

class MertonProblem():

	# Declare the parameters for the problem
	# Initial wealth (W0)
	# Relative risk-aversion (gamma)
	# Distribution parameters for the risky asset (mu, sigma)
	def __init__(self, W0, gamma, mu, sigma):
		self.W0 = W0
		self.gamma = gamma
		self.mu = mu
		self.sigma = sigma

		# Subjective utility discount rate
		self.rho = 0.7
		# Risk-free rate
		self.r = 0.9
		# Number of time steps
		self.T = 1000

		# Current time step
		self.time = 0


    # Utility function for the problem
    # based on constant relative risk-aversion (CRRA)
    def UtilityOfConsumption(self, c):
    	if self.gamma = 1:
    		return log(c)
    	else:
    		return (c^(1-gamma))/(1-gamma) 


    # Closed-form optimal split of the
    def optimalSplit(self):
    	# Declare the optimal policy
    	pi_opt = (self.mu - self.r)/(1-self.gamma)



