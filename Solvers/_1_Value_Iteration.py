# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics

class ValueIteration(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.V = np.zeros(env.nS)

    def train_episode(self):
        """
            What/Where is this function?
                Imagine the for loop from the gym intro (_0_Gym_Introduction.py)
                This function effectively gets called 1 time for each
                loop/iteration of that for loop.
                It is just in a class with more variables and logging
            
            Inputs: (Available/Useful variables)
                self.env
                    this is no different than 
                        import gym
                        self.env = gym.make("CartPole-v1")
                        # from http://gym.openai.com/
                    except that its easier to switch environments
                
                state = self.env.reset():
                    returns a state which is just an integer index
                    (not all problems have state as an index, but this one does)
                    more specifically it resets the environment and returns the stating state
                
                self.env.nS:
                    number of states in the environment
                    
                self.env.nA:
                    number of actions in the environment
                    
                for probability, next_state, reward, done in self.P[state][action]:
                    `probability` will be probability of `next_state` actually being the next state
                    `reward` is the short-term/immediate reward for achieving that next state
                    `done` is a boolean of wether or not that next state is the last/terminal state
                    
                    Every action has a chance (at least theortically) of different outcomes (states)
                    Which is why `self.P[state][action]` is a list of outcomes and not a single outcome
                
                self.options.gamma:
                    The discount factor (gamma from the slides)
            
            Outputs: (what you need to output/update)
                self.V:
                    This should be a dictionary
                    `self.V[state]` should return a floating point value that
                    represents the value of a state. This value should become 
                    more accurate with each episode.
                    
                    How should this be calculated?
                        look at the value iteration algorithm
                        Ref: Sutton book eq. 4.10.
                    Once those values have been updated, thats it for this function/class
        """
        
        # you can add variables here if it is helpful
        
        # Update the estimated value of each state
        for each_state in range(self.env.nS):
            
            ###################################################
            #            Compute self.V here                  #
            # Do a one-step lookahead to find the best action #
            ###################################################


        # Dont worry about this part
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.

        Use:
            self.env.nA: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            """
            What/Where is this function?
                Imagine the gym intro code (_0_Gym_Introduction.py)
                This function is the part that decides what action to take
            
            Inputs: (Available/Useful variables)
                self.V[state]
                    the estimated long-term value of getting to a state
                
                values = self.one_step_lookahead(state)
                    len(values) will be the number of actions (self.env.nA)
                    values[action] will be the expected value of that action (float)
                    
                for probability, next_state, reward, done in self.P[state][action]:
                    `probability` will be probability of `next_state` actually being the next state
                    `reward` is the short-term/immediate reward for achieving that next state
                    `done` is a boolean of wether or not that next state is the last/terminal state
                    
                    Every action has a chance (at least theortically) of different outcomes (states)
                    Which is why `self.P[state][action]` is a list of outcomes and not a single outcome
                
                self.env.nS:
                    number of states in the environment
                    
                self.env.nA:
                    number of actions in the environment
                    
                self.options.gamma:
                    The discount factor (gamma from the slides)
            
            Outputs: (what you need to output)
                return a list
                    len(the_list) should be the number of actions
                    the action we want to take should have a value of 1
                    all other actions should have a value of 0
            """
            
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
        
        return policy_fn


class AsynchVI(ValueIteration):

    def __init__(self,env,options):
        super().__init__(env,options)
        # list of States to be updated by priority
        self.pq = PriorityQueue()
        # A mapping from each state to all states potentially leading to it in a single step
        self.pred = {}
        for s in range(self.env.nS):
            # Do a one-step lookahead to find the best action
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            self.pq.push(s, -abs(self.V[s]-best_action_value))
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def train_episode(self):
        """
        Run a *single* update for Asynchronous Value Iteration Algorithm (using prioritized sweeping).
        Updtae only one state, the one with the highest priority

        Use:
            self.env: OpenAI env. env.P represents the transition probabilities of the environment.
                env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                env.nS is a number of states in the environment.
                env.nA is a number of actions in the environment.
            self.options.gamma: Gamma discount factor.
            self.pred[s]: a list of states leading to state s in one step with probability > 0
            self.pq: list of States to be updated by priority
        """

        #########################################################
        # YOUR IMPLEMENTATION HERE                              #
        # Choose state with the maximal value change potential  #
        # Do a one-step lookahead to find the best action       #
        # Update the value function. Ref: Sutton book eq. 4.10. #
        #########################################################


        # In DP methods we don't interact with the environment so we will set the reward to be the sum of state values
        # and the number of steps to -1 representing an invalid value
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Asynchronous VI"

    def one_step_lookahead(self, state: int):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
