import gym
import numpy as np
import random


def state_memory(S, state):
    if state not in S:
        S.append(state)


def action_memory(A, state, action):
    if action not in A[state]:
        A[state].append(action)


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(Q, state, action, reward, state2, alpha, gamma):
    increment = reward + gamma * np.max(Q[state2]) - Q[state, action]
    Q[state, action] = Q[state, action] + alpha * increment


def Dyna_Q(Q, Model, epsilon, alpha, gamma, total_episodes, max_steps, n_planning):
    S = []
    A = [[] for _ in range(env.observation_space.n)]
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()[0]
        while t < max_steps:
            action = choose_action(state1, epsilon)
            state2, reward, is_truncated, is_finished, prob = env.step(action)
            # If there is a hole, the transition probability of walking into a hole = 0
            # keep the state unchanged
            if is_truncated and reward == 0:
                env.P[state1][action][0] = (1.0, state1, 0.0, False)
                break
            update(Q, state1, action, reward, state2, alpha, gamma)
            Model[state1, action] = [reward, state2]
            state_memory(S, state1)
            action_memory(A, state1, action)
            for i in range(n_planning):
                s = random.choice(S)
                a = random.choice(A[s])
                r, s_next = Model[s][a][0], int(Model[s][a][1])
                update(Q, s, a, r, s_next, alpha, gamma)
            # reach the Terminal state
            if reward == 1 and is_truncated:
                break
            state1 = state2
            t += 1
        epsilon = epsilon - (1 / total_episodes)
    return Q


def get_optimal_policy(Q):
    env_new = gym.make("FrozenLake-v1", render_mode = "human", is_slippery=False)
    state = env_new.reset()[0]
    optimal_policy = []
    while True:
        optimal_action = np.argmax(Q[state, :])
        optimal_policy.append(optimal_action)
        state, reward, is_truncated, is_finished, prob = env_new.step(optimal_action)
        if reward == 1 and is_truncated:
            return optimal_policy


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    epsilon = 1
    alpha = 0.2  # control the increment
    gamma = 0.95  # the decay of reward
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    Model = np.zeros([env.observation_space.n, env.action_space.n, 2])
    Q_value = Dyna_Q(Q, Model, epsilon, alpha, gamma, total_episodes=500, max_steps=50, n_planning=5)
    print("The Q value after Sarsa method")
    print(Q_value)
    optimal_policy = get_optimal_policy(Q_value)
    print("The optimal policy is: ")
    print(optimal_policy)
