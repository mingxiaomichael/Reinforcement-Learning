import numpy as np
import gym
import torch
import os
# from Actor_Critic import Agent
from ac_new import ActorCritic

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    # agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99,
    #               layer1_size=256, layer2_size=256)
    # agent = Agent(input_dims=2, p_fc_dims=[64, 32], v_fc_dims=[64, 32], n_actions=1,
    #               lr_p=0.0005, lr_v=0.0005, gamma=0.99)
    state_dims = env.observation_space.shape[0]
    hidden_dims = 128
    actor_lr = 1e-4
    critic_lr = 1e-4
    gamma = 0.99
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    agent = ActorCritic(state_dims, hidden_dims, 1, actor_lr, critic_lr, gamma, device)

    scores = []
    steps = []
    num_episodes = 200
    for i in range(num_episodes):
        done = False
        score = 0
        step = 0
        state = env.reset()[0]
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        # while (not done) and (step <= 50000):
        while not done:
            action = agent.choose_action(state)
            if action > 1:
                action_e = 1
            if action < -1:
                action_e = -1
            else:
                action_e = action
            next_state, reward, done, info, _ = env.step([action_e])
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            score += reward
            step += 1
            # if step % 100 == 0:
            #     print(step)
        agent.learn(transition_dict)
        scores.append(score)
        steps.append(step)
        print('episode: ', i, 'score: %.2f' % score)

    torch.save(agent.actor.state_dict(), 'actor.pth')
    torch.save(agent.critic.state_dict(), 'actor.pth')

    file_name = "scores.txt"
    with open(file_name, "w") as file:
        for score in scores:
            file.write(f"{score}\n")
    file_name = "steps.txt"
    with open(file_name, "w") as file:
        for step in steps:
            file.write(f"{step}\n")
