import gym
import numpy as np
import random
from gridworld_sv1 import GridworldEnv


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = random.choice(env.action_space)
    else:
        action = np.argmax(Q[state, :])
    return action


def n_step_sarsa(env, Q, n, epsilon, alpha, gamma, total_episodes, max_steps):
    for episode in range(total_episodes):
        s = [env.reset()]
        a = [choose_action(s, epsilon)]
        T = 1000
        t = 0
        r = [0]
        while True:
            if t < T:
                next_s, reward, is_finished = env.step(a[t])
                s.append(next_s)
                r.append(reward)
                if s[t + 1] == env.DONE_LOCATION:
                    T = t + 1
                else:
                    a.append(choose_action(s[t + 1], epsilon))
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += np.power(gamma, i - tau - 1) * r[i]
                if tau + n < T:
                    G += np.power(gamma, n) * Q[s[tau + n]][a[tau + n]]
                Q[s[tau]][a[tau]] += alpha * (G - Q[s[tau]][a[tau]])
            if tau == T - 1:
                print(Q)
                break
            t += 1
    return Q


def get_optimal_policy(Q):
    env_new = GridworldEnv()
    s = env_new.reset()
    optimal_policy = []
    while True:
        optimal_action = np.argmax(Q[s])
        optimal_policy.append(optimal_action)
        s, r, is_finished = env_new.step(optimal_action)
        if is_finished:
            return optimal_policy


if __name__ == "__main__":
    env = GridworldEnv()
    n = 3  # steps of n-step sarsa
    epsilon = 0.8
    alpha = 0.2  # control the increment
    gamma = 0.95  # the decay of reward
    Q = np.zeros([env.nS, env.nA])
    Q_value = n_step_sarsa(env, Q, n, epsilon, alpha, gamma, total_episodes=300, max_steps=100)
    optimal_policy = get_optimal_policy(Q_value)
    print(optimal_policy)

