import gym
import numpy as np


def choose_action(state, epsilon):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def update(Q, state, action, reward, state2, action2, alpha, gamma):
    increment = reward + gamma * Q[state2, action2] - Q[state, action]
    Q[state, action] = Q[state, action] + alpha * increment


def sarsa(Q, epsilon, alpha, gamma, total_episodes, max_steps):
    optimal_policy = []
    for episode in range(total_episodes):
        t = 0
        state1 = env.reset()[0]
        action1 = choose_action(state1, epsilon)
        while t < max_steps:
            policy = []
            policy.append(action1)
            # Visualizing the training
            # env.render()

            # Getting the next state
            state2, reward, is_truncated, is_finished, prob = env.step(action1)

            # Choosing the next action
            action2 = choose_action(state2, epsilon)
            policy.append(action2)

            # Learning the Q-value
            update(Q, state1, action1, reward, state2, action2, alpha, gamma)

            state1 = state2
            action1 = action2

            # Updating the respective values
            t += 1
            if is_truncated:
                break
            # If at the end of learning process
            if is_finished:
                if optimal_policy != policy:
                    optimal_policy = policy
                    break
                else:
                    number_of_loop = episode
                    return optimal_policy, episode


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    epsilon = 0.9
    alpha = 0.85  # control the increment
    gamma = 0.95  # the decay of reward
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    optimal_policy, episode = sarsa(Q, epsilon, alpha, gamma, total_episodes=10000, max_steps=100)
    print(optimal_policy, episode)
    # env.reset()
    # env.render()
    # a = [2, 2, 1, 1, 1, 2]
    #
    # for t in a:
    #     env.render()
    #     # action = env.action_space.sample()
    #     action = t
    #     new_state, reward, is_truncated, is_finished, prob = env.step(action)
    #     print(new_state, reward, is_truncated, is_finished)
