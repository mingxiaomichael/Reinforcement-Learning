import gym
import numpy as np
import torch
from PGBaseline import Agent


if __name__ == "__main__":
    env = gym.make("ALE/Pong-v5")
    input_dims = np.shape(env.reset()[0])
    agent = Agent(gamma=0.95, lr=0.001, lr_v=0.001,
                  input_dims=input_dims, n_actions=env.action_space.n)
    max_episodes = 1000
    scores = []
    for episode in range(max_episodes):
        observation = env.reset()[0]
        rewards = []
        log_probs = []
        state_list = []
        done = False
        step = 0
        while not done:
            action, log_prob = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)
            # change the reward function
            if reward == 1:
                reward = 100.0
            else:
                reward = -1.0
            state_list.append(observation_)
            rewards.append(reward)
            log_probs.append(log_prob)
            scores.append(sum(rewards))
            observation = observation_
            step += 1
            if done:  # REINFORCE with baseline
                agent.learn(state_list, rewards, log_probs)
                print(f'episode: {episode}, score: {sum(rewards)},'
                      f'average_score: {np.mean(scores[-10:])}, total_steps: {step}')

    torch.save(agent.PolicyNet.state_dict(), 'Pong_PolicyNet.pth')
    torch.save(agent.ValueNet.state_dict(), 'Pong_ValueNet.pth')

    env.close()
