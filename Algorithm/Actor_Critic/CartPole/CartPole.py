import numpy as np
import gym
import torch
# from ActorCritic import ActorCritic
from ActorCriticFinish import ActorCritic
if __name__ == '__main__':
    # env = gym.make('CartPole-v1', render_mode="human")
    env = gym.make('CartPole-v1')
    input_dims = env.observation_space.shape[0]
    output_dims = env.action_space.n
    fc_dims = 128
    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.98
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = ActorCritic(input_dims, fc_dims, output_dims, actor_lr, critic_lr, gamma, device)
    scores = []
    steps = []
    num_episodes = 1000
    for i in range(num_episodes):
        # transition_memory = []
        done = False
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        score = 0
        state = env.reset()[0]
        step = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info, _ = env.step(action)
            # transition_memory.append([state, action, reward, next_state, done])
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            score += reward
            step += 1
        # print("transtion_memory:", transition_memory)
        agent.update(transition_dict)
        scores.append(score)
        steps.append(step)
        print('episode: ', i, 'score: %.2f' % score)
