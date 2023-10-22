import numpy as np
import gym
import torch
import os
from Actor_Critic import Agent

if __name__ == '__main__':
    # agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99,
    #               layer1_size=256, layer2_size=256)
    agent = Agent(input_dims=2, p_fc_dims=[64, 32], v_fc_dims=[64, 32], n_actions=1,
                  lr_p=0.0005, lr_v=0.0005, gamma=0.99)

    env = gym.make('MountainCarContinuous-v0')
    scores = []
    steps = []
    num_episodes = 200
    for i in range(num_episodes):
        done = False
        score = 0
        step = 0
        observation = env.reset()[0]
        while (not done) and (step <= 50000):
            action = np.array(agent.choose_action(observation)).reshape((1,))
            observation_, reward, done, info, _ = env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
            step += 1
            if step % 100 == 0:
                print(step)
        scores.append(score)
        steps.append(step)
        print('episode: ', i, 'score: %.2f' % score)

    torch.save(agent.ac.state_dict(), 'ac.pth')
    torch.save(agent.ac_target.state_dict(), 'ac_target.pth')

    file_name = "scores.txt"
    with open(file_name, "w") as file:
        for score in scores:
            file.write(f"{score}\n")
    file_name = "steps.txt"
    with open(file_name, "w") as file:
        for step in steps:
            file.write(f"{step}\n")
