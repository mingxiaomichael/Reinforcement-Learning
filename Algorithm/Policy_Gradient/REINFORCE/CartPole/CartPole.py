import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from PolicyNetwork import Agent

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent(gamma=0.95, lr=0.001,
                  input_dims=[4], n_actions=env.action_space.n)
    max_episodes = 2000
    max_steps = 500
    scores = []
    avg_scores = []
    for episode in range(max_episodes):
        observation = env.reset()[0]
        rewards = []
        log_probs = []
        for step in range(max_steps):
            action, log_prob = agent.choose_action(observation)
            next_observation, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            # start REINFORCE algorithm
            if done:  # Generate a trajectory
                agent.update_policy(rewards, log_probs)
                scores.append(sum(rewards))
                avg_scores.append(np.mean(scores[-50:]))
                print(f'episode: {episode}, score: {sum(rewards)},'
                      f'average_reward: {np.mean(scores[-50:])}, total_steps: {step}')
                break
            observation = next_observation
    torch.save(agent.PolicyNet.state_dict(), 'CartPole_model.pth')

    plt.plot(scores)
    plt.plot(avg_scores)
    plt.legend(['scores', 'avg_scores'])
    plt.xlabel('episode')
    plt.show()
