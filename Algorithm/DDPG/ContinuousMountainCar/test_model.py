from DDPG_torch import Agent
import gym
import numpy as np
import torch as T
import random

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    agent = Agent(alpha=0.0001, beta=0.001, gamma=0.99, input_dims=2, fc_dims=[64, 64], n_actions=1,
                  batch_size=64, tau=0.001, device=device)
    agent.actor.load_state_dict(T.load('actor.pth'))
    scores = []
    for i in range(10):
        state = env.reset()[0]
        done = False
        score = 0
        step = 0
        while not done:
            action = agent.choose_action(state, turn_off_noise=True)
            next_state, reward, done, info, _ = env.step(action)
            score += reward
            state = next_state
        scores.append(score)
        print('episode ', i, 'score %.2f' % score, 'trailing 10 games avg %.3f' % np.mean(scores[-10:]))
