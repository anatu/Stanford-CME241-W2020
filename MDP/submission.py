import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    
    def __init__(self):
        self.N = 7

    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        actions = ["Move"]
        return actions
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        results = []

        if action == "Move":
            if state == self.N:
                return []
            possible_outcomes = self.N - state
            for i in range(state+1, self.N+1):
                if i == self.N - 1:
                    moveReward = 1
                else:
                    moveReward = 100
                probability = 1/possible_outcomes
                results.append((i, probability, moveReward))
        
        return results
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0.99
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        
        # Initialize results vector
        # Should be list of 3-tuples with (newState, probability, and reward)
        results = []

        # Check if the current state is an end state by checking if 
        # the deck is set to None, and return an empty list if so
        def isEnd(state, action):
            return state[2] == None

        if isEnd(state, action):
            return []
        
        # Pull counts of each card type
        # Find the total number of remaining cards to calculate probability
        # of any given card type appearing when we create future states
        ft_counts = state[2]
        ft_sum = sum(ft_counts)
            
        # QUIT ACTION
        # If quit then there's only one option of future state
        if action == "Quit":
            newState = (state[0], None, None)
            results.append((newState, 1, state[0]))

        # TAKE ACTION
        # For d types of cards remaining in deck (d <= n_facevalues), there are d possible
        # future states. (facevalue, None, (...,n_i-1,...))
        if action == "Take":
            # WITH PREVIOUS PEEK
            # If peeked, the next card is determined
            if state[1] != None:
                cardvalue_sum = state[0] + self.cardValues[state[1]]
                # Give reward if drawing the last card, otherwise 0
                if sum(state[2]) == 1:
                    reward = cardvalue_sum
                else:
                    reward = 0
                # If player goes bust with drawn card, end-state + zero reward
                if cardvalue_sum > self.threshold:
                    newState = (cardvalue_sum, None, None)
                    results.append((newState, 1, 0))
                # Otherwise regular take action for the peeked card
                else:
                    new_ft = list(ft_counts)
                    new_ft[state[1]] = new_ft[state[1]] - 1
                    new_ft = tuple(new_ft)
                    newState = (cardvalue_sum, None, new_ft)
                    results.append((newState, 1, reward))
            # WITHOUT PREVIOUS PEEK
            # Iterate through every different type of card
            for i in range(len(ft_counts)):
                if ft_counts[i] > 0:
                    # Calculate the probability of that card occurring
                    quotient = float(ft_counts[i])/ft_sum
                    probability = int(quotient) if quotient.is_integer() else quotient
                    cardvalue_sum = state[0] + self.cardValues[i]
                    # Decrement by 1 the amount of the chosen card being taken
                    # reward is zero if it's not the last card in the deck
                    if ft_sum > 1:
                        new_ft = list(ft_counts)
                        new_ft[i] = new_ft[i] - 1
                        new_ft = tuple(new_ft)
                        reward = 0
                    # END STATE - Last card in deck drawn
                    # which means the game ends and reward is the sum of
                    # card values in hand (if not above threshold, which is 
                    # handled by the check below)
                    else:
                        new_ft = None
                        reward = cardvalue_sum
                    # Sum of values in hand
                    # If over threshold, then the player goes bust
                    if (cardvalue_sum) > self.threshold:
                        newState = (cardvalue_sum, None, None)
                        results.append((newState, probability, 0))
                    # Otherwise, continue
                    # Note that this can still reach an end state if the card drawn
                    # is the last card as per the new_ft check above
                    else:
                        newState = (cardvalue_sum, None, new_ft)
                        results.append((newState, probability, reward))

        # PEEK ACTION
        if action == "Peek":
            # Check if peek has already been done, and return empty list if so
            if state[1] != None:
                return []
            # Otherwise update with possible peek results
            else:
                for i in range(len(ft_counts)):
                    if ft_counts[i] > 0:
                        # Calculate probability of the given card type being the next card
                        quotient = float(ft_counts[i])/ft_sum
                        probability = int(quotient) if quotient.is_integer() else quotient
                        # Create the new state with that card type in the peek element
                        newState = (state[0], i, ft_counts)
                        results.append((newState, probability, -1*self.peekCost))
        return results
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # Make 10% of the cards greater than the threshold so that peeking is required in at least
    # 10% of instances to determine whether taking a card is tractable
    return BlackjackMDP(cardValues = [21, 1, 2], multiplicity = 3, threshold = 20, peekCost = 1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
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

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

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


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
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

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    valiter = ValueIteration()
    valiter.solve(original_mdp)
    orig_pi = valiter.pi
    print valiter.pi

    vi_rl = util.FixedRLAlgorithm(orig_pi)
    vi_result = util.simulate(modified_mdp, vi_rl, 30000, verbose = False)

    orig_rl = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor, explorationProb = 0.2)
    orig_result = util.simulate(modified_mdp, orig_rl, 30000, verbose = False)


    return vi_result, orig_result

    # END_YOUR_CODE
