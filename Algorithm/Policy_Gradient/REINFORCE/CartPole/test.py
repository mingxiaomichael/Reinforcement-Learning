import gym
import torch
from PolicyNetwork import Agent

env_test = gym.make("CartPole-v0", render_mode="human")
agent_test = Agent(gamma=0.95, lr=0.001, input_dims=[4], n_actions=2)
agent_test.PolicyNet.load_state_dict(torch.load('CartPole_model.pth'))
show_episode = 3
score_list = []
for i in range(show_episode):
    observation = env_test.reset()[0]
    score = 0
    done = False
    while not done:
        action, log_prob = agent_test.choose_action(observation)
        next_observation, reward, done, _, _ = env_test.step(action)
        score += reward
        observation = next_observation
        print(next_observation, score, done)
    score_list.append(score)
print(score_list)






