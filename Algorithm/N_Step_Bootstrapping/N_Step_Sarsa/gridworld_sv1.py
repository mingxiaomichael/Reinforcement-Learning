import numpy as np
from gym.envs.toy_text import discrete

"""
How to use the environment 'gridworld_sv0.py':
1. add 'gridworld_sv0.py' to gym package path 'xxxxx\\Lib\\site-packages\\gym\\envs\\toy_text';
2. In the same path, write 'from gym.envs.toy_text.gridworld_sv0 import GridworldEnv' into '__init__.py';
3. register 'gridworld_sv0.py' environment in path 'xxxxx\\Lib\\site-packages\\gym\\envs', add code:
    register(
    id='GridWorld_sv0',
    entry_point='gym.envs.toy_text:GridworldEnv',
    )
"""


class GridworldEnv:
    """
    For example, a 5x5 grid looks as follows:

    s  o  o  x  x
    o  o  o  x  o   # DONE_LOCATION == 24
    x  o  o  o  o   # OBSTACLE == [3, 4, 8, 10, 15, 16]
    x  x  o  o  o
    o  o  o  o  T

    s is start position, x is obstacle and T is the terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    def __init__(self):
        self.shape = [5, 5]
        self.action_space = [0, 1, 2, 3]
        self.UP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        self.DONE_LOCATION = 24
        self.OBSTACLE = [3, 4, 8, 10, 15, 16]

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.MAX_Y = self.shape[0]
        self.MAX_X = self.shape[1]

        self.reward = -1 * np.ones(self.nS)
        self.reward[self.nS - 1] = 100
        for s in range(self.nS):
            if s in self.OBSTACLE:
                self.reward[s] = -5
        # state transition probability
        self.P = {}
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            self.P[s] = {a: [] for a in range(self.nA)}
            is_done = lambda s: s == self.DONE_LOCATION
            # We're stuck in a terminal state
            if is_done(s):
                self.P[s][self.UP] = [(1, s, self.reward[s], True)]
                self.P[s][self.RIGHT] = [(1, s, self.reward[s], True)]
                self.P[s][self.DOWN] = [(1, s, self.reward[s], True)]
                self.P[s][self.LEFT] = [(1, s, self.reward[s], True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - self.MAX_X
                ns_right = s if x == (self.MAX_X - 1) else s + 1
                ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X
                ns_left = s if x == 0 else s - 1
                self.P[s][self.UP] = [(1, ns_up, self.reward[ns_up], is_done(ns_up))]
                self.P[s][self.RIGHT] = [(1, ns_right, self.reward[ns_right], is_done(ns_right))]
                self.P[s][self.DOWN] = [(1, ns_down, self.reward[ns_down], is_done(ns_down))]
                self.P[s][self.LEFT] = [(1, ns_left, self.reward[ns_left], is_done(ns_left))]
            it.iternext()
        # initialize
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # next_state, reward, is_finished
        _, next_state, reward, is_finished = self.P[self.state][action][0]
        self.state = next_state
        return next_state, reward, is_finished
