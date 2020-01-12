import collections, random

# MDP Class Structure, Subclassed for a particular MDP Problem
# (Adapted from earlier coursework for CS221)
class MDP:
    # Return the start state.
    def startState(self):
    	raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): 
    	raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): 
    	raise NotImplementedError("Override me")

    # Sets the discount rate for the problem
    def discount(self): 
    	raise NotImplementedError("Override me")

    # Compute set of states reachable from startState
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
