# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, MaxPooling2D, Softmax, Input
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from skimage.transform import resize
from skimage import color
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


def pg_loss(rewards):
    def loss(labels, predicted_output):
        """
        The policy gradient loss function.

        args:
            deltas: Cumulative discounted rewards.
            labels: True actions (one-hot encoded actions).
            predicted_output: Predicted actions (action probabilities).

        Use:
            K.log: Element-wise log.
            K.mean: Mean of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################


    return loss

keras.losses.pg_loss = pg_loss

class Reinforce(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = (self.env.observation_space.shape[0],)
        self.action_size = self.env.action_space.n
        self.trajectory = []
        self.model = self.build_model()
        self.policy = self.create_greedy_policy()

    def build_model(self):
        rewards = Input(shape=(1,))
        layers = self.options.layers
        states = Input(shape=self.state_size)
        d = states
        for l in layers:
            d = Dense(l, activation='relu')(d)
        do = Dense(self.action_size)(d)
        out = Softmax()(do)

        opt = Adam(lr=self.options.alpha)
        model = Model(inputs=[states, rewards], outputs=out)
        model.compile(optimizer=opt, loss=pg_loss(rewards))
        return model

    def create_greedy_policy(self):
        """
        Creates a greedy policy.
        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            return self.model.predict([[state], np.zeros((1, 1))])[0]

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the REINFORCE algorithm
        Use:
            self.model: Policy network that is being learned.
            self.policy(state): Returns action probabilities.
            self.options.steps: Maximal number of steps per episode.
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in probs.
            self.step(action): Performs an action in the env.
            self.trajectory.append((state, action, next_state, reward)): Add the last
                transition to the observed trajectory (don't forget to reset the
                trajectory at the end of each episode).
            np.zeros_like(): Return an array of zeros with the a given shape.
            self.env.reset(): Resets the env.
            self.options.gamma: Gamma discount factor.
            self.model.fit(): Train the policy network at the end of an episode on the
                transitions in self.trajectory for exactly 1 epoch. Make sure that
                the cumulative rewards are discounted.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################


    def __str__(self):
        return "REINFORCE"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
