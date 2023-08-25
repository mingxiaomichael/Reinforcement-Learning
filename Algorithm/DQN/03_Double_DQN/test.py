import gym

env = gym.make('CartPole-v1', render_mode='human')

num_episodes = 5
max_steps_per_episode = 10
observation = env.reset()[0]
print(observation)

for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    state = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        env.render()

        action = env.action_space.sample()
        action[0] = round(action[0], 1)

        next_state, reward, _, _, _ = env.step(action)

        total_reward += reward
        state = next_state

        print(f"Step {step + 1}/{max_steps_per_episode} - Action: {action}, State: {next_state}, Reward: {reward}")
env.close()

# import gymnasium as gym
# import numpy as np
#
# env = gym.make('CartPole-v1', g=9.81)
# print("action_space: ", env.action_space.n)
# print("observation_space: ", env.observation_space.n)
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

    # scores.append(score)