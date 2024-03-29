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

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DONE_LOCATION = 8


class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For ContinuousMountainCar, a 5x5 grid looks as follows:

    o  o  o  o  o
    o  o  o  T  o   # DONE_LOCATION == 8
    o  o  o  o  o
    o  x  o  o  o
    o  o  o  o  o

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    def __init__(self, shape=[5, 5]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == DONE_LOCATION
            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1, s, reward, True)]
                P[s][RIGHT] = [(1, s, reward, True)]
                P[s][DOWN] = [(1, s, reward, True)]
                P[s][LEFT] = [(1, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)
