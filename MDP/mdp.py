import util, math, random
from collections import defaultdict
from util import ValueIteration



# Sample MDP which subclasses the utility MDP data structure
# using a simple example
# The set of possible states in this game is {-2, -1, 0, 1, 2}. You start at state 0, and if you reach either -2 or 2, the game ends. At each state, you can take one of two actions: {-1, +1}.
# If you're in state s and choose -1:
# You have an 80% chance of reaching the state s−1.
# You have a 20% chance of reaching the state s+1.
# If you're in state s and choose +1:
# You have a 70% chance of reaching the state s+1.
# You have a 30% chance of reaching the state s−1.
# If your action results in transitioning to state -2, then you receive a reward of 20. If your action results in transitioning to state 2, then your reward is 100. Otherwise, your reward is -5. Assume the discount factor γ is 1.
class SampleMDP(util.MDP):

	# Don't need a constructor here  
    # def __init__(self):

    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
    	# Start at State 0
        return 0

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        actions = ["-1", "+1"]
        return actions

    def isTerminalState(self, state):
    	return (state == 2) or (state == -2)

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        results = []
        # Check for terminal states
        if isTerminalState:
        	return []
    	if state == 1:
    		fwdReward = 100
    	else:
    		fwdReward = -5
   		if state == -1:
   			backReward = 20
   		else:
   			backReward = -5 	
        if action == "+1":
        	results.append((state+1,0.7,fwdReward))
        	results.append((state-1,0.3,backReward))

        elif action == "-1":
        	results.append((state+1,0.2,fwdReward))
        	results.append((state-1,0.8,backReward))        
        return results

    # Set the discount factor (float or integer) to discount future rewards
    def discount(self):
        return 1



############################################################

# A simple example of an MDP where states are integers in [-n, +n].
# and actions involve moving left and right by one position.
# We get rewarded for going to the right.
class NumberLineMDP(MDP):
    def __init__(self, n=5): self.n = n
    def startState(self): return 0
    def actions(self, state): return [-1, +1]
    def succAndProbReward(self, state, action):
        return [(state, 0.4, 0),
                (min(max(state + action, -self.n), +self.n), 0.6, state)]
    def discount(self): return 0.9

############################################################


# class Lecture2MRP(util.MRP)
# 	def __init__:
		

# 	def startState(self):
# 		return 0
