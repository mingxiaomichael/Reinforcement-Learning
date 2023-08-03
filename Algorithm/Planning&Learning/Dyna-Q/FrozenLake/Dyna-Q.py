import gym
import numpy as np


def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(Q, state, action, reward, state2, action2, alpha, gamma):
    increment = reward + gamma * Q[state2, action2] - Q[state, action]
    Q[state, action] = Q[state, action] + alpha * increment


def sarsa(Q, epsilon, alpha, gamma, total_episodes, max_steps):
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()[0]
        action1 = choose_action(state1, epsilon)
        while t < max_steps:
            # Getting the next state
            state2, reward, is_truncated, is_finished, prob = env.step(action1)
            # If there is a hole, the transition probability of walking into a hole = 0
            # keep the state unchanged
            if is_truncated and reward == 0:
                env.P[state1][action1][0] = (1.0, state1, 0.0, False)
                break
            # Choosing the next action
            action2 = choose_action(state2, epsilon)
            update(Q, state1, action1, reward, state2, action2, alpha, gamma)
            state1 = state2
            action1 = action2
            t += 1
            # (for test)If reaching the terminal state
            # if reward == 1 and is_truncated:
            #     print(episode)
            #     print(Q)
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
    epsilon = 0.9
    alpha = 0.2  # control the increment
    gamma = 0.95  # the decay of reward
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    Q_value = sarsa(Q, epsilon, alpha, gamma, total_episodes=500, max_steps=50)
    print("The Q value after Sarsa method")
    print(Q_value)
    optimal_policy = get_optimal_policy(Q_value)
    print("The optimal policy is: ")
    print(optimal_policy)
