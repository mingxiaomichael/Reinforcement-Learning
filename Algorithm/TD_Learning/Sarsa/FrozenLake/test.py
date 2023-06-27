"""
this python file is just for test, not useful :)
"""
import gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)

env.reset()
env.render()
# a = [2, 2, 1, 1, 1, 2]
a = [1, 1, 2, 1, 2, 2]
for t in a:
    env.render()
    # action = env.action_space.sample()
    action = t
    new_state, reward, is_truncated, is_finished, prob = env.step(action)
    print(new_state, reward, is_truncated, is_finished)
