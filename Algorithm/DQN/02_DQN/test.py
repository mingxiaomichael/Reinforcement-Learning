import gym
import numpy as np

env = gym.make("LunarLander-v2", render_mode="human")
max_episodes = 1
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
        print(j, score, reward, done)
        if done:
            # print(state, reward, done)
            # done == True, indicates reach the terminal goal or the truncated state
            # how to solve: making the score higher before done == True
            if reward != -100:
                print(state, reward, done)

    scores.append(score)