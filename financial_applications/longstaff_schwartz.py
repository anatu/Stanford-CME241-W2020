# Adapted from https://github.com/AlexYH-Chen/Path-Dependant-Option-Pricing/blob/master/Option%20Pricing.ipynb
# and https://github.com/ganesh-k-sahu/LSMC_Longstaff_Swartz/blob/master/simple_american_option.py

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




if __name__ == "__main__":


	# Initialize mxn price matrix SP for m monte carlo paths 
	# over n+1 time steps
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

    M, N = SP.shape
    STRIKE_PRICE = 1.1
    RISKLESS_RATE = 0.06

    # Initialize the results matrix
    option_cash_flow_matrix = np.zeros((M,N))





