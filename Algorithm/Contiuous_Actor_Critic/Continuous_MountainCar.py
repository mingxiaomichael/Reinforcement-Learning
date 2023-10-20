import numpy as np
import gym
import torch
from Actor_Critic import Agent


if __name__ == '__main__':
    agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99,
                  layer1_size=256, layer2_size=256)

    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    scores = []
    num_episodes = 100
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()[0]
        while not done:
            action = np.array(agent.choose_action(observation)).reshape((1,))
            observation_, reward, done, info, _ = env.step(action)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        scores.append(score)
        print('episode: ', i, 'score: %.2f' % score)

    torch.save(agent.actor.state_dict(), 'Actor_model.pth')
    torch.save(agent.critic.state_dict(), 'Critic_model.pth')

    file_name = "scores.txt"
    with open(file_name, "w") as file:
        for score in scores:
            file.write(f"{score}\n")
