import gym
import math
import numpy as np
from Double_DQN import Agent
from DrawAverageScore import DrawAverageScore


def exponential_decay(initial, i, decay_rate=0.01):
    """
    Being used for decaying 'lr' and 'epsilon'
    """
    return initial * math.exp(-decay_rate * i)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    action_space = np.round(np.arange(-2, 2.01, 0.1), 1)
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=[3], batch_size=64,
                  n_actions=len(action_space))
    n_episodes = 400
    n_learn_steps = 100
    initial_epsilon = agent.epsilon
    scores, average_scores = [], []
    for i in range(n_episodes):
        score = 0
        observation = env.reset()[0]
        j = 0
        while j < n_learn_steps:
            action = agent.choose_action(observation)
            observation_, reward, _, _, _ = env.step([action_space[action]])
            score += reward
            agent.store_transition(observation, action, reward, observation_)
            agent.learn()
            observation = observation_
            j += 1
        agent.epsilon = exponential_decay(initial_epsilon, i)
        scores.append(score)
        avg_score = np.mean(scores[-20:])
        average_scores.append(avg_score)
        print('episode:', i, 'score: %.2f' % score,
              'average score: %.2f' % avg_score, 'epsilon: %.2f' % agent.epsilon)

    DrawAverageScore(n_episodes, average_scores)

    # Start Test
    agent_test = agent
    env_new = gym.make("CartPole-v1", render_mode="human")
    show_episodes = 20
    show_steps = 100
    agent_test.epsilon = 0
    for i in range(1, show_episodes + 1):
        observation = env_new.reset()[0]
        j = 0
        while j < show_steps:
            env_new.render()
            action = agent_test.choose_action(observation)
            observation_, reward, _, _, _ = env_new.step([action_space[action]])
            env_new.render()
            observation = observation_
            j += 1
