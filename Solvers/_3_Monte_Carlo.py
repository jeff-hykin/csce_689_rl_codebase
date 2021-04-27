# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from collections import defaultdict, OrderedDict
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting
#env = Blackjack-v0


class MonteCarlo(AbstractSolver):

    def __init__(self,env,options):
        assert (str(env.observation_space).startswith('Discrete') or
        str(env.observation_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete state spaces"
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        # Add required fields          #
        # You are free to add any      #
        # attribute you deem necessary.#
        # Just make sure that you don't#
        # store all the episode for    #
        # calculating the Q-function.  #
        # Use a moving average over    #
        # Q-values instead.            #
        ################################



    def train_episode(self):
        """
            Run a single episode for Monte Carlo Control using Epsilon-Greedy policies.

            Use:
                self.options.env: OpenAI gym environment.
                self.options.gamma: Gamma discount factor.
                self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
                new_state, reward, done, _ = self.step(action): To advance one step in the environment

            Note:
                train_episode is called multiple times from run.py. Within
                train_episode you need to store the transitions in 1 complete
                trajectory/episode. Then using the transitions in that episode,
                update the Q-function. You should NOT store multiple episodes.
                You should update the Q-function using a moving average instead
                to take into account the Q-values from the previous episodes.
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################


    def __str__(self):
        return "Monte Carlo"

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-estimates and epsilon.

        Use:
            self.Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA
            self.options.epsilon: Chance the sample a random action. Float betwen 0 and 1.
            self.env.action_space.n: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """
        nA = self.env.action_space.n

        def policy_fn(observation):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################


        return policy_fn

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################


        return policy_fn

    def plot(self,stats):
        # For plotting: Create value function from action-value function
        # by picking the best action at each state
        V = defaultdict(float)
        for state, actions in self.Q.items():
            action_value = np.max(actions)
            V[state] = action_value
        plotting.plot_value_function(V, title="Final Value Function")


class OffPolicyMC(MonteCarlo):
    def __init__(self,env,options):
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)

        # The cumulative denominator of the weighted importance sampling formula
        # (across all episodes)
        self.C = defaultdict(lambda: np.zeros(env.action_space.n))

        # Our greedily policy we want to learn about
        self.target_policy = self.create_greedy_policy()
        # Our behavior policy we want to learn from
        self.behavior_policy = self.create_random_policy()

    def train_episode(self):
        """
            Run a single episode of Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.

            Use:
                self.options.env: OpenAI gym environment.
                self.options.gamma: Gamma discount factor.
                new_state, reward, done, _ = self.step(action): To advance one step in the environment
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################


    def create_random_policy(self):
        """
        Creates a random policy function.

        Use:
            self.env.action_space.n: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) / nA

        def policy_fn(observation):
            return A

        return policy_fn

    def __str__(self):
        return "MC+IS"
