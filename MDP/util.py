import collections, random
import numpy as np

############################################################
############################################################
# ITERATION ALGORITHMS
############################################################
############################################################

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
        print "ValueIteration: %d iterations" % numIters
        self.pi = pi
        self.V = V


class BellmanMatrix(MDPAlgorithm):
    '''
    Solves an MRP using the Bellman equation using a direct method
    by matrix inversion of the system V = R + gamma*P*V
    so that the optimal value function is computed analytically
    as (I-gamma*P)^(-1)*R using matrix inversion / algebra.
    
    V is a vector of value functions for states 1...n
    R is rewards for states 1...n
    P is the transition probability matrix P_ij = prob of moving from state i to j    
    gamma is discount rate
    '''

    def solve(self, mrp):
        gamma = mrp.discount()

        # Pull states from utility function
        states = mrp.computeStates()

        # Initialize matrices
        R = np.zeros(len(states))
        P = np.zeros((len(states),len(states)))
        I = np.eye(len(states))

        rewardDict = mrp.stateRewards() 

        for i in range(len(states)):
            state = states(i)
            R[i] = rewardDict[state]
            successors = mrp.succAndProbReward(state)
            # (newState, prob, reward) tuples
            for succ in successors:
                succState = succ[0]
                prob = succ[1]
                P[i][states.index(succState)] = prob
        
        V = np.matmul(np.linalg.inv(I-gamma*P),R)        

        return V

############################################################
############################################################
# MARKOV CHAIN DATA STRUCTURES
############################################################
############################################################

# An abstract class representing a Markov Decision Process (MDP). 
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")


    def discount(self): raise NotImplementedError("Override me")


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


# An abstract class representing a Markov Reward Process (MRP). 
class MRP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")


    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, s'), reward = Reward(s, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Return a dict mapping states to rewards
    # Additional helper function for algorithm to quickly pull the reward for each unique state
    # rather than having to parse it out from the computeStates function
    def stateRewards(self, state, action): raise NotImplementedError("Override me")

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
            for newState, prob, reward in self.succAndProbReward(state):
                if newState not in self.states:
                    self.states.add(newState)
                    self.rewards.add()
                    queue.append(newState)
        # print "%d states" % len(self.states)
        # print self.states





############################################################
############################################################
# RL ALGORITHMS AND FIXED-POLICY EVALUATION
############################################################
########################################################################################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRLAlgorithm(RLAlgorithm):
    def __init__(self, pi): self.pi = pi

    # Just return the action given by the policy.
    def getAction(self, state): return self.pi[state]

    # Don't do anything: just stare off into space.
    def incorporateFeedback(self, state, action, reward, newState): pass


# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        Q_current = self.getQ(state, action)

        new_actions = self.actions(newState)
        new_Qs = []
        for act in new_actions:
            new_Qs.append(self.getQ(newState, act))
        V_opt = max(new_Qs)

        features = self.featureExtractor(state, action)
        stepSize = self.getStepSize()
        for feature in features:
            key = feature[0]
            value = feature[1]
            self.weights[key] = self.weights.get(key, 0) - stepSize*(Q_current - (reward + self.discount*V_opt))*value    

############################################################



############################################################
############################################################
# SIMULATION
############################################################
############################################################
# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            action = rl.getAction(state)
            transitions = mdp.succAndProbReward(state, action)
            if sort: transitions = sorted(transitions)
            if len(transitions) == 0:
                rl.incorporateFeedback(state, action, 0, None)
                break

            # Choose a random transition
            i = sample([prob for newState, prob, reward in transitions])
            newState, prob, reward = transitions[i]
            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)
            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount()
            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards




def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    valiter = ValueIteration()
    valiter.solve(smallMDP)
    # Simulate with 20% exploration probability, and then set to 0 after simulation
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, explorationProb = 0.2)
    util.simulate(mdp, rl, 30000, verbose = False)
    rl.explorationProb = 0
    # Extract the optimal policies and replicate the dict that comes from valiter.pi
    rl_result = dict()
    same = 0
    different = 0
    for state in valiter.pi.keys():
        rl_result[state] = rl.getAction(state)
        print rl.getAction(state), valiter.pi[state]
        if rl.getAction(state) == valiter.pi[state]:
            same = same + 1
        else: 
            different = different + 1

    print same, different

    return valiter.pi, rl_result    
    # END_YOUR_CODE


def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    features = []
    # Indicator for action and current total
    features.append(((action, total), 1))

    if counts != None:
        count_id = []
        for i in range(len(counts)):
            # Indicator for counts of each card
            features.append(((action, i, counts[i]), 1))
            if counts[i] == 0: count_id.append(0)
            else: count_id.append(1)
        # Indicator for the action and presence/absence of each face value
        features.append(((action, tuple(count_id)), 1))
    return features
    # END_YOUR_CODE