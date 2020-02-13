import os
import re
import sys
sys.path.append('../')
from MDP import util
import numpy as np
import matplotlib.pyplot as plt


###########################################################
# Value Iteration Algorithm
###########################################################
# An algorithm that solves an MDP (i.e., computes the optimal
# policy).
class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")

############################################################
# Instantiate these classes, then call the solve function on the MDP object
# corresponding to the MDP that you want to solve
class ValueIteration(MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print("ValueIteration: %d iterations" % numIters)
        self.pi = pi
        self.V = V


###########################################################
# Problem 1 - MDP Setup
###########################################################
class Problem1MDP(util.MDP):
    
    def __init__(self, N):
        self.N = N

    # Return the start state.
    def startState(self):
        # Start in the middle of the pond (floor-divide)
        return self.N//2

    # Return set of actions possible from |state|.
    def actions(self, state):
        actions = ["A","B"]
        return actions

    def isEnd(self, state):
        result = (state == 0) or (state == self.N)
        return result

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action):
        results = []
        
        if self.isEnd(state):
            return []

        if action == "A":
            succ1 = state - 1
            if succ1 == 0:
                reward1 = -10
            else:
                reward1 = -1
            results.append((succ1, state/self.N, reward1))

            succ2 = state + 1
            if succ2 == self.N:
                reward2 = 100
            else:
                reward2 = -1
            results.append((succ2, (self.N-state)/self.N, reward2))
        elif action == "B":
            for i in range(self.N+1):
                succ = i
                if succ == 0:
                    reward = -10
                elif succ == self.N:
                    reward = 100
                else:
                    reward = -1
                results.append((succ, 1/self.N, reward))
        return results


    def discount(self):
        return 1


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
###########################################################
###########################################################


###########################################################
# Problem 2 - MDP Setup (Code Sketch)
###########################################################
class Problem2MDP(util.MDP):

    def __init__(self, jobWages, unempWage, alpha):\
        # List of wages for the possible jobs
        self.jobWages = wageList
        self.jobLabels = ["W{}".format(i) for i in range(len(jobWages))]
        # Float: Wage of unemployment
        self.unempWage = unempWage
        # Float: Probability of unemployment at end of every day
        self.alpha = alpha

    # Return the start state.
    def startState(self):
        # Start Unemployed
        return "Unemployed"

    # Return set of actions possible from |state|.
    def actions(self, state):
        actions = ["Find Job","Wait"]
        return actions


    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action):
        results = []
        if state == "Unemployed":
            # Do not take any job, stay unemployed        
            if action == "Wait":
                results.append(("Unemployed", 1, self.unempWage))
            else:
                # Collect one of the available jobs  (assume they are assigned with equal probability)
                for i in range(len(self.jobWages)):
                    results.append((self.jobLabels[i], 1/len(self.jobWages), self.jobWages[i]))
        else:
            # Stay employed
            results.append((state, 1-self.alpha, self.jobWages[self.jobLabels.index(state)]))
            # Become unemployed (but end of day, so you still collect the day's wage)
            results.append(("Unemployed", self.alpha, self.jobWages[self.jobLabels.index(state)]))
        return results

    # Choose an arbitrary discount rate for this problem
    def discount(self):
        return 0.7


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
###########################################################
###########################################################


if __name__ == "__main__":
    ###########################################################
    # Problem 1 - Value Iteration Solution & Plots
    ###########################################################
    N_vals = [3,10,25]
    vi = util.ValueIteration()
    for n in N_vals:
        print("Running solution for N={}".format(n))
        mdp = Problem1MDP(N=n)
        vi.solve(mdp)
        print(vi.pi)
        print(vi.V)
        Vlist = []
        for key, value in vi.V.items():
            Vlist.append(value)
        plt.plot(list(range(n+1)), Vlist)
        plt.show()
    ###########################################################
    ###########################################################

    ###########################################################
    # Problem 2 - Numerical Solution (Code Sketch)
    ###########################################################
    vi = util.ValueIteration()
    mdp = Problem2MDP(jobWages = [10, 15, 20, 30], unempWage = 5,
        alpha = 0.3)
        vi.solve(mdp)
        print(vi.pi)
        print(vi.V)
    ###########################################################
    ###########################################################
    