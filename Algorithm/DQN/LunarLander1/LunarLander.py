import gym
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
# print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)
# state = env.reset()
# # env.render()
# # env.step(1)
# # env.render()
# for t in range(500):
#     env.render()
#     print(state)
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     state, reward, done, _ = env.step(action)
# env.close()


max_episodes = 10
max_steps = 100
scores = []
actions = range(env.action_space.n)
for i in range(1, max_episodes + 1):
    state = env.reset()
    score = 0
    for j in range(max_steps):
        env.render()
        action = np.random.choice(actions)
        state, reward, done, _, _ = env.step(action)
        env.render()
        score += reward
        print(score)
        # if done:
        #     if i % 20 == 0:
        #         print('Episode {},  score: {}'.format(i, score))
        #     break

    scores.append(score)

# for i in range(5):
#     score = 0
#     state = env.reset()
#     while True:
#         action = dqn_agent.act(state)
#         next_state, reward, done, info = env.step(action)
#         state = next_state
#         score += reward
#         if done:
#             break
#     print('episode: {} scored {}'.format(i, score))

