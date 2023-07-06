import numpy as np
import random

"""
This file tests double Q-learning algorithm on left-right environment (sutton book 6.7 Example 6.7).
"""


# design the class of left-right environment
class Env:
    def __init__(self, mu, sigma, nB):
        self.mu = mu
        self.sigma = sigma
        self.STATE_A = self.left = 0
        self.STATE_B = self.right = 1
        self.Terminal = 2
        self.nS = 3
        self.nA = 2
        self.nB = nB  # the number of state of B, if choosing left
        self.state = self.STATE_A  # initialize

    def reset(self):
        self.state = self.STATE_A  # start in state A
        return self.state

    def step(self, action):
        # state: A  action: right
        if self.state == self.STATE_A and action == self.left:
            self.state = self.STATE_B
            return self.state, 0, False  # next_state, reward, is_finished
        # state: A  action: right
        elif self.state == self.STATE_A and action == self.right:
            self.state = self.Terminal
            return self.state, 0, True
        # state: B  action: all
        elif self.state == self.STATE_B:
            self.state = self.Terminal
            reward = random.normalvariate(self.mu, self.sigma)
            return self.state, reward, True

