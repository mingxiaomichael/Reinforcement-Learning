import gym
import numpy as np


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(Q, s, a, r, s_next, a_next, alpha, gamma):
    increment = r + gamma * Q[s_next, a_next] - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * increment


def sarsa(Q, epsilon, alpha, gamma, total_episodes, max_steps):
    for episode in range(total_episodes):
        t = 0
        s = env.reset()[0]
        a = choose_action(s, epsilon)
        while t < max_steps:
            # Getting the next state
            s_next, r, is_finished, _, prob = env.step(a)
            # Choosing the next action
            a_next = choose_action(s_next, epsilon)
            update(Q, s, a, r, s_next, a_next, alpha, gamma)
            s = s_next
            a = a_next
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
