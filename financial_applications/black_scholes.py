# Implementation of Black-Scholes model for derivatives pricing / hedging in incomplete and complete markets
import math
import datetime
from datetime import timedelta
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats
import sys
sys.path.append('../')
from MDP import util

# Math adapted from https://www.investopedia.com/terms/b/blackscholes.asp
# https://medium.com/swlh/black-scholes-algorithmic-delta-hedging-c2cdd42ce175
# https://www.coursera.org/lecture/reinforcement-learning-in-finance/mdp-formulation-b8ZIT

'''
Model for pricing and hedging of derivatives in complete and incomplete markets
The complete market assumption means no transaction costs, and a price for every asset (i.e. one can always transact
any asset class at any point in time subject to its FMV)
'''


'''State = 
t = current point in time
Alpha = Hedge positions (units held) at time t
P = prices per unit of hedges at time t
Beta = PnL position at time t
D = Portfolio state amongst m derivatives


Action = a (units of hedges traded at price P)
'''

'''
S_t = Stock Price
Define X = -(mu-sig^2/2) + log (S_t) (Brownian motion)
action variable u(S) = a(X(S))
'''



class HedgePricing:

	def __init__(self) -> None:
		'''
		Initialize parameters for the problem
		'''
		self.T = 100
		self.t = 0
		self.alpha = 0
		self.beta = 0
		self.D = 1

	def PnL_update(self, a) -> None:
		'''
		Update the P&L position for each time step
		Beta_new = Beta_old Plus...
		+ Cashflows from holding positions (X + alpha*Y)
		- Cost of units of hedges purchased (a*P)
		- Transaction costs (e.g. -gamma*P*abs(a), we will use this)
		'''
		# Evolve price and return data (TODO: Where do we get a simulator for this?)
		self.X = np.random.normal(0,1)
		self.Y = np.random.normal(5,1)
		self.P = np.random.normal(4,3)

		# Update the portfolio
		self.beta = self.beta + self.X + (self.alpha*self*Y) - (a*self.P) - (self.gamma*self.P*abs(a))


	def utilityFunc(self, beta) -> float:
		'''
		Calculates reward as utility of consumption from the
		existing portfolio state
		'''
		# CRRA coefficient (relative risk-aversion)
		coeff = 0.5
		result = np.exp(beta^(1-coeff) - 1)/(1-coeff)
		return result

	def constructMDP(self):
		mdpData = dict()
		
		# Observe state
		state = (self.t, self.alpha, self.P, self.beta, self.D)
		
		# 
