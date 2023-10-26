from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('MountainCarContinuous-v0')
# agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[2], tau=0.001, env=env,
#               batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)
agent = Agent(alpha=0.001, beta=0.001, input_dims=[2], tau=0.01, env=env,
              batch_size=512,  layer1_size=64, layer2_size=64, n_actions=1)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(200):
    obs = env.reset()[0]
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info, _ = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
    score_history.append(score)

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-10:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)