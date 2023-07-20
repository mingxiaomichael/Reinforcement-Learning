import gym
import numpy as np
from DQN import Agent


if __name__ == "__main__":
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[8], lr=0.003)
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)


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



