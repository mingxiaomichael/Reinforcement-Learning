import gym
import torch as T
import numpy as np
from DDPG_model import DDPG

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    agent = DDPG(alpha=0.0001, beta=0.001, gamma=0.99, input_dims=env.observation_space.shape[0], fc_dims=[64, 64],
                 n_actions=1, tau=0.001, device=device)
    np.random.seed(0)
    scores = []
    for i in range(200):
        state = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            agent.replay_buffer.store_transition(state, action, reward, next_state, int(done))
            agent.learn()
            score += reward
            state = next_state
        scores.append(score)
        print('episode: ', i, 'score: %.2f' % score, '10 games average %.3f' % np.mean(scores[-10:]))
