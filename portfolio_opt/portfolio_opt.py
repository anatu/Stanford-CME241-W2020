import os
import re
import sys
sys.path.append('../')
from MDP import util
import numpy as np

# See https://en.wikipedia.org/wiki/Merton%27s_portfolio_problem

class MertonProblem():

	# Declare the parameters for the problem
	# Initial wealth (W0)
	# Relative risk-aversion (gamma)
	# Distribution parameters for the risky asset (mean and variance)
	def __init__(self, W0, gamma, mu, sigma):
		self.W0 = W0
		self.gamma = gamma
		self.mu = mu
		self.var = var

		# Subjective utility discount rate
		self.rho = 0.7
		# Risk-free rate
		self.r = 0.9
		# Total life time
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


    # Closed-form optimal solution derived analytically
    def closedFormSolution(self):
    	# Declare epsilon parameter
    	eps = 1e-6

    	# Optimal split fraction between risky and riskless asset
    	pi_opt = (self.mu - self.r)/(self.var*self.gamma)

    	# Compute parameter
    	nu = (self.rho/self.gamma) - (1-self.gamma)*(((self.mu-self.r)*pi_opt/2*self.gamma)+(self.r/self.gamma))

    	# Compute the wealth fraction
    	if self.time == self.T:
    		c_opt = nu
    	elif nu == 0:
    		c_opt = (self.T-self.time-eps)^(-1)
    	else:
    		c_opt = (nu*(1+(nu*eps-1)*np.exp(-nu*(self.T-self.time)))^(-1)




if __name__ == "main":
	mp = MertonProblem()
	