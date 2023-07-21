import gym
import numpy as np
from DQN import Agent


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    # env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 300
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()[0]
        j = 0
        while (not done) and (j < 200):
            # env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            # env.render()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            j += 1
        agent.epsilon = agent.epsilon - agent.eps_dec if agent.epsilon > agent.epsilon_min else agent.epsilon_min
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-50:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)

    env_new = gym.make("LunarLander-v2", render_mode="human")
    show_episode = 50
    for i in range(1, show_episode + 1):
        observation = env_new.reset()[0]
        done = False
        while not done:
            env_new.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env_new.step(action)
            env_new.render()
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_


# max_episodes = 1
# max_steps = 100
# scores = []
# actions = range(env.action_space.n)
# for i in range(1, max_episodes + 1):
#     state = env.reset()
#     score = 0
#     for j in range(max_steps):
#         env.render()
#         action = np.random.choice(actions)
#         state, reward, done, _, _ = env.step(action)
#         env.render()
#         score += reward
#         print(j, score, reward, done)
#         if done:
#             # print(state, reward, done)
#             # done == True, indicates reach the terminal goal or the truncated state
#             # how to solve: making the score higher before done == True
#             if reward != -100:
#                 print(state, reward, done)
#
#     scores.append(score)



