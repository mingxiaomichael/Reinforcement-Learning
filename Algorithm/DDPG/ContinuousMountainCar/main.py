from DDPG_torch import Agent
import gym
import numpy as np
import torch as T
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    agent = Agent(alpha=0.0001, beta=0.001, gamma=0.99, input_dims=2, fc_dims=[64, 64], n_actions=1,
                  batch_size=64, tau=0.001, device=device)

    scores = []
    for i in range(100):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            turn_off_noise = False if i < 80 else True
            action = agent.choose_action(state, turn_off_noise)
            next_state, reward, done, info, _ = env.step(action)
            agent.remember(state, action, reward, next_state, int(done))
            agent.learn()
            score += reward
            state = next_state
        scores.append(score)
        print('episode ', i, 'score %.2f' % score, 'trailing 10 games avg %.3f' % np.mean(scores[-10:]))

    filename = 'plot/average scores2.png'
    plotLearning(scores, filename, window=100)

    T.save(agent.actor.state_dict(), 'actor.pth')

    file_name = "scores.txt"
    with open(file_name, "w") as file:
        for score in scores:
            file.write(f"{score}\n")
