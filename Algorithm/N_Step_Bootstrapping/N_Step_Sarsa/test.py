import random
import numpy as np
from gridworld_sv1 import GridworldEnv

env = GridworldEnv()
p = env.P
state = env.reset()

a = [1, 2, 2, 1, 1, 1, 2, 2]
for i in a:
    next_state, reward, is_finished = env.step(i)
    print(next_state, reward, is_finished)

print(env.nS, env.nA)

b = [0, 1, 2, 3]
print(random.choice(b))

print(float('inf'))
print(np.min(1,2))
