# Adapted from https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/monte_carlo_epsilon_greedy_demo.ipynb

import numpy as np
import matplotlib.pyplot as plt
from gridWorldGame import standard_grid, negative_grid,print_values, print_policy

# Initialize parameters
tol = 1e-4
gamma = 0.9
actions = ("U", "D", "L", "R")


def random_action(a, eps=0.1):
  '''
  Utility function for choosing a random action
  with probability (1-eps) + (eps/N_ACTIONS)
  '''
  p = np.random.random()
  if p < (1 - eps):
    return a
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def max_dict(d):
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



def policy_rollout(grid, policy, mode):
	
