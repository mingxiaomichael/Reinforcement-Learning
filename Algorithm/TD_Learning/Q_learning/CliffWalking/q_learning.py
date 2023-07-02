import gym
import numpy as np


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(Q, s, a, r, s_next, alpha, gamma):
    increment = r + gamma * np.max(Q[s_next]) - Q[s][a]
    Q[s][a] += alpha * increment


def q_learning(Q, epsilon, alpha, gamma, total_episodes, max_steps):
    for episode in range(total_episodes):  # Loop for each episode
        s = env.reset()[0]  # Initialize s
        t = 0
        while t < max_steps:
            a = choose_action(s, epsilon)  # Choose a from s from Q by using eps-greedy
            s_next, r, is_finished, _, prob = env.step(a)  # Take a, observe r, s_next
            update(Q, s, a, r, s_next, alpha, gamma)
            s = s_next
            t += 1
            if r == -100:
                break
            if is_finished:
                print(Q)
                break
    return Q


def get_optimal_policy(Q):
    env_new = gym.make("CliffWalking-v0", render_mode="human")
    s = env_new.reset()[0]
    optimal_policy = []
    while True:
        optimal_action = np.argmax(Q[s, :])
        optimal_policy.append(optimal_action)
        s, r, is_finished, _, prob = env_new.step(optimal_action)
        if is_finished:
            return optimal_policy


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    env.P[35][2][0] = (1.0, 47, 0, True)  # set the reward of the goal to 0
    epsilon = 0.9
    alpha = 0.9  # control the increment
    gamma = 0.95  # the decay of reward
    Q = np.zeros([env.nS, env.nA])
    Q_value = q_learning(Q, epsilon, alpha, gamma, total_episodes=1000, max_steps=500)
    optimal_policy = get_optimal_policy(Q_value)
    print(optimal_policy)
