import os
import re
import sys
import numpy as np
from typing import TypeVar, NamedTuple, Callable, Mapping, Tuple


sys.path.append('../')
from MDP.mdp import MDP
import MDP.mdpUtils as mu
import MDP.mdpAlgorithms as ma

P = TypeVar('P')

'''
FINANCIAL APPLICATION #1 - Merton's Portfolio Optimization Problem
See e.g. https://en.wikipedia.org/wiki/Merton%27s_portfolio_problem


FROM CLASS NOTES:
Think of this as a continuous-time Stochastic Control problem
The State is (t, Wt)
The Action is [πt, ct]
The Reward per unit time is U(ct)
The Return is the usual accumulated discounted Reward
Find Policy : (t, Wt) → [πt, ct] that maximizes the Expected Return
Note: ct ≥ 0, but πt is unconstrained
'''
    

class MertonProblem():

	def __init__(self, **params: Mapping[P, float]) -> None:
       '''
       Constructor. Declare the parameters that we will be using
       to solve this portfolio optimization model 
       ''' 
       self.T: float = T # Expiry value
       self.rho: float = rho # Discount rate
       self.r: float = r # Riskless rate (return for the riskless asset)

       self.W0: float = W0 # Initial wealth
       self.mu: np.ndarray = mu # Means of the risky rate (1-D array of length = # of Risky Assets)
       self.cov: np.ndarray = cov # Risky rate covariants (2-D square array of length = # of Risky Assets)

       self.epsilon: float = epsilon # Bequest parameter for B(T)
       self.gamma: float = gamma # Parameter for CRRA utility function

       self.SIM_TIME = SIM_TIME # Total simulation time

    def utilityFunc(self, x: float) -> float:
        '''
        Utility function for the problem. For Merton's problem we use the 
        Constant Relative-Risk Aversion model represented by the gamma param
        '''
        p = 1. - self.gamma
        if p == 0:
            result = np.log(x)
        else:
            result = x**p/p
        return result

    def getCFOptAllocation(self) -> np.ndarray:
        '''
        Calculates the closed-form solution of the optimal
        stock allocation among the different assets, π(W,t).
        Note however that the function takes no arguments since in the closed-form
        the optimal allocation depends on neither Wealth nor time
        '''
        return np.linalg.inv(self.cov).dot(self.mu-self.r)/self.gamma


    def getNu(self) -> float:
        '''
        Helper method to calculate the nu-parameter used in 
        the defintion of the closed-form solution
        '''
        t1 = (self.mu - self.r).dot(self.getCFOptAllocation())/2*self.gamma
        t2 = self.r/self.gamma
        return (self.rho/self.gamma) - self.gamma*(t1+t2)


    def getCFOptConsumption(self, t: float) -> float: 
        '''
        Calculates the closed-form solution of the optimal fraction
        of wealth to consume at a given time step
        '''
        nu = self.getNu()

        if nu == 0:
            optCons = 1./(self.expiry - t + self.epsilon)
        else:
            optCons = nu / (1. + (nu * self.epsilon - 1) *
                             np.exp(-nu * (self.expiry - t)))            
        return optCons


    def makeDiscretizedMDP(self) -> MDP:
        '''
        Takes the information prescribed for this Merton object
        and builds an MDP out of it.
        '''
        mdpData = dict()
        states = set()
        actions = set()

        # First, define the set of states. We will form an upper bound on wealth as start wealth + 5X stdev of the risky asset
        # over the course of the simulation time. Choose a step size to discretize W over
        wStep = 0.01
        Wmax = self.W0 + (5*self.cov)*self.SIM_TIME

        # Now, iterate over all of the possible state combinations
        for w in range(0,Wmax,wStep):
            for t in range(self.SIM_TIME)
                states.add((w, t))

        # Now, discretize / define the action space using some finite discretization
        # step size similar to the above. 
        aStep = 0.01
        for a in range(0, 1, aStep):
            actions.add(a)



if __name__ == "__main__":
    # Declare params for a simple case with one risky asset
    # and one riskless asset
    params = dict()
    params["T"] = 0.4
    params["rho"] = 0.04
    params["r"] = 0.04

    params["W0"] = 1.
    params["mu"] = np.array([0.08])
    params["cov"] = np.array([0.0009])

    params["epsilon"] = 1e-8
    params["gamma"] = 0.2

    params["SIM_TIME"] = 5



    # Instantiate the object
    mp = MertonProblem(params)

    # TODO: Move this into a member function within the class
    # First, calculate the closed-form solution over our time range
    # for both allocation (constant) and consumption (return the optimal
    # consumption fraction for each timestep)
    optAlloc = mp.getCFOptAllocation()
    optCons = [mp.getCFOptConsumption(t*mp.T/SIM_TIME) for t in range(SIM_TIME)]

    
    
