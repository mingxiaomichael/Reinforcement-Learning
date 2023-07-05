import gym
import numpy as np


"""
Expected Sarsa algorithm uses the expected value of Q(S', A') to update Q(S, A)
In this file, I use epsilon-greedy policy to calculate the expected value of Q(S', A')
"""


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(Q, s, a, r, s_next, alpha, gamma, epsilon):
    # if epsilon here was 0, the Q-table is as same as Q-table of Q-learning
    expected_Q_value = (epsilon / env.nA) * (np.sum(Q[s_next])) + (1 - epsilon) * Q[s_next][np.argmax(Q[s_next])]
    increment = r + gamma * expected_Q_value - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * increment


def sarsa(Q, epsilon, alpha, gamma, total_episodes, max_steps):
    for episode in range(total_episodes):
        t = 0
        s = env.reset()[0]
        a = choose_action(s, epsilon)
        while t < max_steps:
            a = choose_action(s, epsilon)  # Choose a from s from Q by using eps-greedy
            s_next, r, is_finished, _, prob = env.step(a)
            update(Q, s, a, r, s_next, alpha, gamma, epsilon)
            s = s_next
            t += 1
            if r == -100:
                break
            if is_finished:
                print(Q)
                break
        print(Q)
    return Q


def get_optimal_policy(Q):
    env_new = gym.make("CliffWalking-v0", render_mode="human")
    s = env_new.reset()[0]
    optimal_policy = []
    while True:
        if s == 0:  # top left corner, only consider right and down
            optimal_action = 1 if Q[s][1] >= Q[s][2] else 2
        elif s == (env.nS / env.nA) - 1:  # top except two corners, only consider right, down and left
            optimal_action = 2 if Q[s][2] >= Q[s][3] else 3
        elif s in range(1, int(env.nS / env.nA)):  # top right corner, only consider left and down
            optimal_action = np.argmax(Q[s][1:4]) + 1
        else:
            optimal_action = np.argmax(Q[s])
        optimal_policy.append(optimal_action)
        s, r, is_finished, _, prob = env_new.step(optimal_action)
        if is_finished:
            return optimal_policy


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    env.P[35][2][0] = (1.0, 47, 100, True)  # set the reward of the goal to 0
    epsilon = 0.9
    alpha = 0.15  # control the increment
    gamma = 0.95  # the decay of reward
    Q = np.zeros([env.nS, env.nA])
    Q_value = sarsa(Q, epsilon, alpha, gamma, total_episodes=1000, max_steps=500)
    optimal_policy = get_optimal_policy(Q_value)
    print(optimal_policy)
