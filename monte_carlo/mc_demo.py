'''Monte-Carlo example using Grid World game
Here, we are only demonstrating the Monte-Carlo method of system identification
i.e. we will end up with empirical estimates of the value function for every state on the board
HOWEVER, we are not yet optimizing so we are not computing the OPTIMAL value function, and the policy
is never changed.
'''

import numpy as np
from gridWorld import standard_grid, negative_grid,print_values, print_policy

# Initialize parameters
tol = 1e-4
gamma = 0.9
actions = ("U", "D", "L", "R")


def policy_rollout(grid, policy, mode):
	'''
	Rolls out an episode of the game using a given policy,
	and then goes in reverse to calculate the rewards realized over the course of
	the visited state tree to compute cumulative rewards at each state 
	'''
	if mode not in ["monte_carlo", "td"]:
		raise ValueError("Choose either monte_carlo or td method of reward-target calculation")

	# Initialize the agent
	startStates = list(grid.actions.keys())
	grid.set_state(startStates[np.random.choice(len(startStates))])


	# Initialize the state information
	state = grid.current_state()
	succAndReward = [(state,0)]

	# FORWARD PASS
	# Rollout to completion of an episode the game
	while not grid.game_over():
		action = policy[state]
		reward = grid.move(action)
		state = grid.current_state()
		succAndReward.append((state, reward))

	# BACKWARDS PASS
	# Revisit our record of states and rewards to calculate the cumulative reward
	# for the episode

	# For Monte-Carlo we estimate using the actual cumulative reward (G_t)
	if mode == "monte_carlo":
		G = 0
		cumRewards = []	
		firstEntry = True
		# Iterate in reverse, i.e. from the end working back to the start
		for state, reward in reversed(succAndReward):
			# Ignore the terminal state, since it has no value function
			if firstEntry:
				firstEntry = False
			else:
				cumRewards.append((state, G))		
			# Update cumulative reward
			G = reward + gamma*G
		cumRewards.reverse()
	# For TD we need to preserve memory of the individual rewards
	elif mode == "td":
		cumRewards = []	
		firstEntry = True
		# Iterate in reverse, i.e. from the end working back to the start
		for state, reward in reversed(succAndReward):
			# Ignore the terminal state, since it has no value function
			if firstEntry:
				firstEntry = False
			else:
				# Store the individual rewards this time since they 
				# for the basis of our TD target (along with the value function estimate)
				# instead of the cumulative reward G.
				cumRewards.append((state, reward))		
		cumRewards.reverse()


	return cumRewards


def samplePolicy():
	'''
	Provide some random policy to pass to the game
	'''
	policy = {
	(2, 0): 'U',
	(1, 0): 'U',
	(0, 0): 'R',
	(0, 1): 'R',
	(0, 2): 'R',
	(1, 2): 'R',
	(2, 1): 'R',
	(2, 2): 'R',
	(2, 3): 'U',
	}
	return policy

def simulate(grid, policy, T, mode):
	'''
	Simulate a run-through of MC policy evaluation
	for T rollouts of the game, starting with a given policy
	and using either first-visit or every-visit value function estimation
	'''

	if mode not in ["first_visit", "every_visit"]:
		raise ValueError("Algorithm not understood. Must be either first_visit or every_visit")


	# Initialize the value-function data structure for all states
	# and empty lists to stack cumulative rewards into
	# (Initializes the value-function to zero everywhere)
	V = {}
	returns = {}
	states = grid.all_states()

	for state in states:
		if state in grid.actions:
			returns[state] = []
		else:
			V[state] = 0 

	print("Initial Values:")
	print_values(V, grid)

	for t in range(T):
		# Rollout an episode
		cumRewards = policy_rollout(grid, policy, "monte_carlo")
		seenStates = set()

		if mode == "first_visit":
			for state,G in cumRewards:
				# "First-visit" evaluation - only calculate the value function the first time
				# the state is visited in the episode
				if state not in seenStates:
					# Append to list of returns for that state
					returns[state].append(G)
					# Update the value function as the expectation of reward from that state
					V[state] = np.mean(returns[state])
					seenStates.add(state)
		elif mode == "every_visit":
			for state,G in cumRewards:
				# "Every-visit" evaluation - update rewards and calculate the value function
				# every time we visit
				# Append to list of returns for that state
				returns[state].append(G)
				# Update the value function as the expectation of reward from that state
				V[state] = np.mean(returns[state])


	print("Final Values:")
	print_values(V, grid)
	print("Final Policy:")
	print_policy(policy, grid)


def simulate_td(grid, policy, T):
	'''
	Simulate a run-through of Temporal-Difference (TD) policy evaluation
	for T rollouts of the game, starting with a given policy
	'''

	# Initialize the value-function data structure for all states
	# and empty lists to stack cumulative rewards into
	# (Initializes the value-function to zero everywhere)

	# For TD method, our rewards are no longer the cumulative reward itself (G_t) 
	# but instead the reward and discounted value-function at the next timestep (R_t+1 + gamma*V_t+1) 
	V = {}
	returns = {}
	states = grid.all_states()

	for state in states:
		if state in grid.actions:
			returns[state] = []
		else:
			V[state] = 0 

	print("Initial Values:")
	print_values(V, grid)

	for t in range(T):
		# Rollout an episode
		rewards = policy_rollout(grid, policy, "td")


		for i in range(len(rewards)-1):
			# Pull current and future state / reward pairs
			s, r = rewards[i]
			sp, rp = rewards[i+1]

			# Append the target metric to list of returns for that state
			try:
				target = rp + gamma*V[sp]
			except KeyError:
				target = rp 
			returns[s].append(target)
			# Update the value function as the expectation of reward from that state
			V[s] = np.mean(returns[s])


	print("Final Values:")
	print_values(V, grid)
	print("Final Policy:")
	print_policy(policy, grid)




if __name__ == "__main__":
	grid = standard_grid()
	print("Rewards:")
	print_values(grid.rewards, grid)
	policy = samplePolicy()
	print("FIRST-VISIT RESULTS")
	simulate(grid = grid,policy = policy, T=100, mode="first_visit")
	print("EVERY-VISIT RESULTS")
	simulate(grid = grid,policy = policy, T=100, mode="every_visit")

	print("TD METHOD RESULTS")
	simulate_td(grid = grid,policy = policy, T=100)
