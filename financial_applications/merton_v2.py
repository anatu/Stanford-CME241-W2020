import os
import re
import sys
import numpy as np
from typing import TypeVar, NamedTuple, Callable, Mapping, Tuple


sys.path.append('../')
sys.path.append('../MDP')
from MDP.mdp import MDP, MDP_RL, Policy
import MDP.mdpUtils as mu
from MDP.rlAlgorithms import MDPAlgorithmRL, MonteCarloEG
from MDP.mdpAlgorithms import MDPAlgorithm, ValueIteration 


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

    def __init__(self, **kwargs) -> None:
        '''
        Constructor. Declare the parameters that we will be using
        to solve this portfolio optimization model 
        ''' 

        self.rho: float = kwargs["rho"] # Discount rate
        self.T: float = kwargs["T"] # Expiry value
        self.r: float = kwargs["r"] # Riskless rate (return for the riskless asset)

        self.W0: float = kwargs["W0"] # Initial wealth
        self.mu: np.ndarray = kwargs["mu"] # Means of the risky rate (1-D array of length = # of Risky Assets)
        self.cov: np.ndarray = kwargs["cov"] # Risky rate covariants (2-D square array of length = # of Risky Assets)
        self.numRiskyAssets = len(self.mu) # The number of risky assets that we have defined for the problem

        self.epsilon: float = kwargs["epsilon"] # Bequest parameter for B(T)
        self.gamma: float = kwargs["gamma"] # Parameter for CRRA utility function

        self.SIM_TIME = kwargs["SIM_TIME"] # Total simulation time

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
        # Handle the 1D case without matrix operations
        if self.cov.shape == (1,1):
            result = (1/self.cov[0][0])*(self.mu[0]-self.r)/self.gamma
        else:
            result = np.linalg.inv(self.cov).dot(self.mu-self.r)/self.gamma
        return result


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
            optCons = 1./(self.T - t + self.epsilon)
        else:
            optCons = nu / (1. + (nu * self.epsilon - 1) *
                             np.exp(-nu * (self.T - t)))            
        return optCons


    def makeDiscretizedMDP(self) -> MDP:
        '''
        Takes the information prescribed for this Merton object
        and builds a discretized MDP out of it so that it can be solved
        using dynamic programming.

        To do this we will make several simplifying assumptions:
        - Discretize the state- and action-space so we can solve using a tabular method
        - Since we can't model stochastic dynamics we will make the risky asset only have two possible
        outcomes (+mu and -mu), with 50/50 probability. This assumption lets us directly prescribe
        transition probabilities into the data structure 
        
        '''
        mdpData = dict()
        states = set()
        actions = set()
        terminalStates = set()

        # Discretize the state- and action-spaces
        # State-space discretization. We will discretize only up to
        # the max limit of growth of wealth by 5 stdevs of the highest value
        # of risky asset return over the total simulation time
        wStep = 0.01
        Wmax = round(self.W0 + (5*(np.max(self.cov)+np.max(self.mu))*self.SIM_TIME),2)
        N = (Wmax // wStep) + 2
        w = 0

        for w in np.linspace(0.,Wmax,N):
            for t in range(self.SIM_TIME+1):
                if t == self.SIM_TIME:
                    terminalStates.add((round(w,2), t))
                else:
                    states.add((round(w,2), t))

        # Action-space discretization
        actionStep = 0.01
        aN = (1//actionStep) + 1
        for pi in np.linspace(0, 1, aN):
            for c in np.linspace(0, 1, aN):
                actions.add((pi,c))

        # Fill out the dict with our discretized state- and action-spaces. 
        for state in states:
            mdpData[state] = dict()
            for action in actions:
                W, t = state
                pi, c = action
                mdpData[state][action] = dict()

                # Determine the two possible successor states, for each of the possible
                # two return values of the risky asset.
                # the net of our returns from all assets, less the amount we invested (c_t*W_t)
                # (Note that is NOT the same as reward, which is computed using the utility function) 
                wPos = round(W + W*c*((1-pi)*self.r + pi*self.mu[0]), 2)                                
                wNeg = round(W + W*c*((1-pi)*self.r - pi*self.mu[0]), 2)                                
                # Assign reward of the final wealth if it's a terminal state. Otherwise
                # the reward is zero
                if t == self.SIM_TIME:
                    tNew = t + 1
                    rPos = self.utilityFunc(wPos)
                    rNeg = self.utilityFunc(wNeg)
                else:
                    tNew = t
                    rPos = 0
                    rNeg = 0

                succPos = (wPos, tNew)
                succNeg = (wNeg, tNew)
                # Populate dict
                if state not in terminalStates:
                    mdpData[state][action][succPos] = (0.5, rPos) 
                    mdpData[state][action][succNeg] = (0.5, rNeg) 
                else:
                    mdpData[state][action][state] = (1.0, 0)

        return MDP(mdpData, (1-self.rho))


    def makeRLMDP(self) -> MDP_RL:
        '''
        Helper method used to transform the parameter definitions for this Merton problem
        into an MDP representation that can be solved using RL. Here we will prescribe a
        state-action dict, a dynamics model, and a set of terminal states.
        '''
        states = set()
        terminalStates = set()
        actions = set()
        stateActionDict = dict()

        # Discretize the state- and action-spaces
        # State-space discretization. We will discretize only up to
        # the max limit of growth of wealth by 5 stdevs of the highest value
        # of the risky asset return realized for each step of the total simulation time
        wStep = 0.01
        Wmax = round(self.W0 + (5*(np.max(self.cov)+np.max(self.mu))*self.SIM_TIME),2)
        N = (Wmax // wStep) + 2
        w = 0
        print(Wmax)
        for w in np.linspace(0.,Wmax,N):
            for t in range(self.SIM_TIME+1):
                if t == self.SIM_TIME:
                    terminalStates.add((round(w,2), t))
                else:
                    states.add((round(w,2), t))
        print(states)

        # Action-space discretization
        actionStep = 0.01
        aN = (1//actionStep) + 1
        for pi in np.linspace(0, 1, aN):
            for c in np.linspace(0, 1, aN):
                actions.add((pi,c))

        # Form the state-action dictionary
        for state in states:
            stateActionDict[state] = set()
            for action in actions:
                stateActionDict[state].add(action)


        # Declare the dynamics model to increment the state
        def dynamics(state, action):
            W, t = state
            pi, c = action

            # Increment time forward
            tNew = t + 1

            # Increment wealth forward based on action
            wNew = W + W*c*((1-pi)*self.r + pi*np.sum(np.random.multivariate_normal(self.mu, self.cov, 1)))                                
            wNew = round(wNew, 2)

            # Form the new successor state
            succState = (wNew, tNew)

            # If we move into a terminal state, calculate the terminal utility
            # as reward. Otherwise, reward is zero
            if tNew == self.SIM_TIME:
                reward = self.utilityFunc(wNew)
            else:
                reward = 0

            return succState, reward

        return MDP_RL(stateActionDict, terminalStates, dynamics)


if __name__ == "__main__":
    # Declare params for a simple case with one risky asset
    # and one riskless asset
    params = dict()
    params["T"] = 0.4
    params["rho"] = 0.04
    params["r"] = 0.04

    params["W0"] = 1.00
    params["mu"] = np.array([0.08])
    params["cov"] = np.array([[0.0009]])

    params["epsilon"] = 1e-8
    params["gamma"] = 0.2

    params["SIM_TIME"] = 5

    # Instantiate the object
    mp = MertonProblem(**params)

    # First, calculate the closed-form solution over our time range
    # for both allocation (constant) and consumption (return the optimal
    # consumption fraction for each timestep)
    optAlloc = mp.getCFOptAllocation()
    optCons = [mp.getCFOptConsumption(t*mp.T/mp.SIM_TIME) for t in range(mp.SIM_TIME)]
    # print(optAlloc) 
    # print(optCons) 

    # RL Solution
    ############################################
    # # Create the RL MDP representation
    # rl_mdp = mp.makeRLMDP()

    # # Make a random policy out of the info
    # polData = dict()
    # for state in rl_mdp.stateActionDict.keys():
    #     polData[state] = dict()
    #     numActions = len(rl_mdp.stateActionDict[state])
    #     for action in rl_mdp.stateActionDict[state]:
    #         polData[state][action] = 1/numActions
    # startPol = Policy(polData)
    
    # # Instantiate the MC solver
    # mc_eg = MonteCarloEG(rl_mdp, 1-params["rho"])
    
    # startState = (params["W0"], 0)

    # # Run simulation and perform model-free policy iteration
    # # using EG policy improvement
    # policy, optValue = mc_eg.simulate_eg(startState, startPol, 100)

    # # Report the results
    # for state in policy.polData.keys():
    #     maxAct, _ = mu.maximizeOverDict(policy.polData[state])
    #     print(maxAct)

    ############################################


    # Discrete MDP solution
    ############################################
    # Create the MDP representation
    mdp = mp.makeDiscretizedMDP()

    # Solve using value iteration
    vi = ValueIteration(1e-8)
    vi.solve(mdp)

    print(vi.V)
    print(vi.pi)


