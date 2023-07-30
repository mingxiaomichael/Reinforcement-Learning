import gym
import math
import numpy as np
from DQN import Agent


def exponential_decay(x, i, decay_rate=0.02):
    """
    Being used for decaying 'lr' and 'epsilon'
    """
    x = x * math.exp(-decay_rate * i)
    return x


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  input_dims=[8], lr=0.003)
    n_episodes = 300
    n_learn_steps = 200
    scores, eps_history = [], []
    for i in range(n_episodes):
        score = 0
        done = False
        observation = env.reset()[0]
        j = 0
        while (not done) and (j < n_learn_steps):
            # env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            # env.render()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            j += 1
        # agent.epsilon = agent.epsilon - agent.eps_dec if agent.epsilon > agent.epsilon_min else agent.epsilon_min
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-50:])
        print('episode:', i, 'score: %.2f' % score,
              'average score: %.2f' % avg_score, 'epsilon: %.2f' % agent.epsilon)

    # Start Test
    env_new = gym.make("LunarLander-v2", render_mode="human")
    show_episodes = 50
    show_steps = 300
    for i in range(1, show_episodes + 1):
        observation = env_new.reset()[0]
        done = False
        j = 0
        while (not done) and (j < show_steps):
            env_new.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env_new.step(action)
            env_new.render()
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            j += 1
