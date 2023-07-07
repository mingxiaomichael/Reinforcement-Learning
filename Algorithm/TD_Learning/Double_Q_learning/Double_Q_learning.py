import numpy as np
import random
from tqdm import tqdm
from LeftPlot import show_figure

"""
This file tests double Q-learning algorithm on left-right environment (sutton book 6.7 Example 6.7).
"""


# design the class of left-right environment
class Env:
    def __init__(self, mu, sigma, nB):
        self.mu = mu
        self.sigma = sigma
        self.STATE_A = self.left = 0
        self.STATE_B = self.right = 1
        self.Terminal = 2
        self.nS = 3
        self.nA = 2
        self.nB = nB  # the number of state of B, if choosing left
        self.state = self.STATE_A  # initialize

    def reset(self):
        self.state = self.STATE_A  # start in state A
        return self.state

    def step(self, action):
        # state: A  action: right
        if self.state == self.STATE_A and action == self.left:
            self.state = self.STATE_B
            return self.state, 0, False  # next_state, reward, is_finished
        # state: A  action: right
        elif self.state == self.STATE_A and action == self.right:
            self.state = self.Terminal
            return self.state, 0, True
        # state: B  action: all
        elif self.state == self.STATE_B:
            self.state = self.Terminal
            reward = random.normalvariate(self.mu, self.sigma)
            return self.state, reward, True


# Q-table
# Because the number of actions per state is different
# Using dictionary is better
def init_Q_table(env):
    Q = {env.STATE_A: {action: 0 for action in range(env.nA)},
         env.STATE_B: {action: 0 for action in range(env.nB)},
         env.Terminal: {action: 0 for action in range(env.nA)}}
    return Q


# epsilon-greedy policy
def select_action_behavior_policy(action_value_dict, epsilon):
    if np.random.uniform(0, 1) < epsilon:  # np.random.uniform(0, 1) include 0 but not include 1
        action = random.sample(list(action_value_dict.keys()), 1)[0]  # random.sample() returns a list
    else:
        max_keys = [key for key, value in action_value_dict.items() if value == max(action_value_dict.values())]
        action = random.choice(max_keys)
    return action


# return Q1+Q2
def get_Q1_add_Q2(Q1_state_dict, Q2_state_dict):
    return {action: Q1_value + Q2_state_dict[action] for action, Q1_value in Q1_state_dict.items()}


def double_Q_learning(env, alpha=0.2, epsilon_scope=[0.2,0.05,0.99], num_of_episode=1000, gamma=0.9):
    epsilon = epsilon_scope[0]
    A_left = np.zeros(num_of_episode)
    Q1 = init_Q_table(env)
    Q2 = init_Q_table(env)
    for num in range(num_of_episode):
        state = env.reset()
        while True:
            add_Q1_Q2_state = get_Q1_add_Q2(Q1[state], Q2[state])
            action = select_action_behavior_policy(add_Q1_Q2_state, epsilon)
            next_state, reward, is_finished = env.step(action)
            if state == env.STATE_A and action == env.left:
                A_left[int(num)] += 1
            if random.random() >= 0.5:
                A1 = random.choice([action for action, value in Q1[next_state].items() if value == max(Q1[next_state].values())])
                Q1[state][action] += alpha * (reward + gamma * Q2[next_state][A1] - Q1[state][action])
            else:
                A2 = random.choice([action for action, value in Q2[next_state].items() if value == max(Q2[next_state].values())])
                Q2[state][action] += alpha * (reward + gamma * Q1[next_state][A2] - Q2[state][action])
            if is_finished:
                break
            state = next_state
            # decaying epsilon
            if epsilon >= epsilon_scope[1]:
                epsilon *= epsilon_scope[2]
    Q = {}
    for state in range(env.nS):
        Q[state] = get_Q1_add_Q2(Q1[state], Q2[state])
    return Q, A_left


if __name__ == "__main__":
    total_num = 100
    prob_Q_A_left = np.zeros((total_num, 300))
    env = Env(-0.1, 1, 10)
    Q_table = double_Q_learning(env, alpha=0.2, epsilon_scope=[0.2, 0.05, 0.99], num_of_episode=300, gamma=0.9)
    for num in tqdm(range(total_num)):
        _, A_left = double_Q_learning(env, alpha=0.2, epsilon_scope=[0.2, 0.05, 0.99], num_of_episode=300, gamma=0.9)
        prob_Q_A_left[int(num)] = A_left
    a = prob_Q_A_left.mean(axis=0)
    show_figure(a)
