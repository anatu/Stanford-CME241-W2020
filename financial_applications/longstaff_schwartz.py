# Adapted from https://github.com/AlexYH-Chen/Path-Dependant-Option-Pricing/blob/master/Option%20Pricing.ipynb
# and https://github.com/ganesh-k-sahu/LSMC_Longstaff_Swartz/blob/master/simple_american_option.py

# http://web.stanford.edu/class/cme241/lecture_slides/AmericanOptionsRL.pdf

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import laguerre
from scipy.interpolate import interp1d
from scipy.stats import norm
import copy
from sklearn.linear_model import LinearRegression


def optionPayoff(s, x):
	'''
	Calculates the payoff of an option with
	fair-market value x purchased at strike price s
	'''
	return max((s-x), 0.0)


def discountCF(t,r,size):
	'''
	Produces the discount rate vector used to update the
	option cash flow matrix at a given time-horizon t (i.e. the time away from terminal) 
	using some riskless rate r, for given vector size
	'''
	mult = [np.exp(-r*(i+1)*t) for i in range(size)]
	mult.reverse()
	return np.array(mult)

def basisFunction(prices):
	'''
	Basis or feature function for the state (i.e. a set of prices)
	which serves as the predictor in our regression (X). Here we will simply
	use a quadratic basis function (i.e. phi(x) = x**2). Accepts / returns a vector
	'''
	return np.power(prices, 2)



if __name__ == "__main__":


	# Declare price matrix
	SP = np.array([
	    [1.00,	1.09, 1.08,	1.34],
	    [1.00, 1.16, 1.26, 1.54],
	    [1.00, 1.22, 1.07, 1.03],
	    [1.00, 0.93, 0.97, 0.92],
	    [1.00, 1.11, 1.56, 1.54],
	    [1.00, 0.76, 0.77, 0.9],
	    [1.00, 0.92, 0.84, 1.01],
	    [1.00, 0.88, 1.22, 1.34]
	    ])


	# Pull shape variables (note that SP is actually m x (n-1) to be consistent with other notation)
    M, N = SP.shape
    STRIKE_PRICE = 1.1
    RISKLESS_RATE = 0.06

    # Vectorize the payoff function so we can directly call it on 
    # vector slices
    payoffVec = np.vectorize(optionPayoff)

    # Initialize the cash flow matrix with payoffs from the last column of SP
    cashFlows = np.zeros((M,N))
    lastPrice = SP[:,-1]
    cashFlows[:, -1] = payoffVec(lastPrice)

    # Iterate
    for t in range(N):
    	# Extract prices for this time step
    	prices = SP[:,j]

    	# Calculate option payoffs from these prices
    	payoffs = payoffVec(prices)

    	# Calculate the discount factor for future payoffs
    	discount = discountCF(j, RISKLESS_RATE, M)

    	# Calculate discounted option cash-flows
    	discountedCashFlows = discount*cashFlows

    	# Sum across time to create 1D vector which has PV of all payoffs
    	CF = np.sum(discountedCashFlows, axis=1)
    	CF = np.array([CF]).transpose()

    	# Calculate features for the prices
    	basisPrices = basisFunction(prices)

    	# Stack the relevant data together so we can filter out the positive-payoff entries altogether
    	stack = np.hstack([payoffs, CF, prices, basisPrices])
    	# Filter the entire stack only for entries where payoffs are positive
    	nonZeroPayoffs = stack[stack[:,0] != 0]

    	# Form X and Y and perform linear regression
    	X = nonZeroPayoffs[:, 3]
    	Y = nonZeroPayoffs[:, 1]
    	reg = LinearRegression().fit(X,Y)

        # Use the regression equation to calculate the continuation value 
        regInput = np.hstack([prices, basisPrices])
        contValue = reg.predict(regInput)

        # Now make a new stack of relevant vectors for the option payoffs,
        # as well as the continuation value, and create a mask to filter steps where
        # payoff exceeds conditional expected continuation value 
        stack2 = np.hstack([payoffs, contValue, payoffs > contValue])

        # Mask out the rows which have a payoff of zero
        stack2[stack2[:,0] == 0, 2] = 0

        

    	













