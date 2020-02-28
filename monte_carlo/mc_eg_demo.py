# Adapted from https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/monte_carlo_epsilon_greedy_demo.ipynb

import numpy as np
import matplotlib.pyplot as plt
from gridWorldGame import standard_grid, negative_grid,print_values, print_policy

# Initialize parameters
tol = 1e-4
gamma = 0.9
actions = ("U", "D", "L", "R")


def chooseRandomAction(a, eps=0.1):
  '''
  Utility function for choosing a random action
  with probability (1-eps) + (eps/N_ACTIONS)
  '''
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(actions)

def maximizeOverDict(d):
  '''
  Utility function to find the optimal value and argument
  from a state tree. Since our state tree is represented as a dict,
  we iterate through keys and pull max key / max value
  '''
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val



def policy_rollout(grid, policy):
	'''
	Perform a rollout of the game subject to a given policy. 
	This will be very similar to the non-EG version, except now
	we will have to explicitly account for actions taken by the agent
	such that we can use the epsilon-greedy technique to improve the policy
	'''
	
	# Initialize the agent
	startState = (2,0)
	grid.set_state(startState)
	startAction = chooseRandomAction(policy[s])

	# Now our tuples are (s, a, r)
	# Initialize memory with the starting state-action pair, and no reward
	succActionReward = []
	succActionReward.append((startState, startAction, 0))


	# FORWARD PASS
	while True:
		reward = grid.move(action)
		state = grid.current_state()

		# Store the (s,a,r) tuple but store
		# no action if we have reached a terminal state
		if grid.game_over():
			succActionReward.append((state, None, reward))
			break
		else:
			# Choose an action stochastically around the action that is
			# prescribed by the policy
			action = chooseRandomAction(policy[state])
			succActionReward.append((state, action, reward))
			
	# BACKWARDS PASS
	G = 0
	firstEntry = True
	cumRewards = []

	for state, action, reward in reversed(succActionReward):

		# Store the cumulative reward at each step workign backwards,
		# but do not store the tuple from the terminal state
		if firstEntry:
			firstEntry = False
		else:
			cumRewards.append((state, action, G))
		# Update the cumulative reward G
		G = reward + gamma*G

	cumRewards.reverse()

	return cumRewards


if __name__ == "__main__":
	grid = negative_grid(step_cost=-0.1)
	# print rewards
	print("rewards:")
	print_values(grid.rewards, grid)

	# state -> action
	# initialize a random policy
	policy = {}
	for s in grid.actions.keys():
	  policy[s] = np.random.choice(actions)
	  
	# initial policy
	print("initial policy:")
	print_policy(policy, grid)




