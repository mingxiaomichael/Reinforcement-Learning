import gym
import numpy as np
from PolicyPlot import draw_policy
"""
This 'ValueIteration.py' file uses gym model environment, 'gridworld_sv0.py'
\\RL\\Reinforcement-Learning\\Algorithm\\PolicyIteration\\Gridworld\\gridworld_sv0.py
"""


def value_iteration(env, discount_factor=1.0, threshold=0.00001):
    def get_max_index(action_values):
        indices = []
        policy_arr = np.zeros(len(action_values))
        action_max_value = np.max(action_values)
        for i in range(len(action_values)):
            action_value = action_values[i]
            if action_value == action_max_value:
                indices.append(i)
                policy_arr[i] = 1
        return indices, policy_arr

    def action_value_function(s, v):  # calculate q(s,a) (action value function)
        q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                q[a] += (reward + discount_factor * prob * v[next_state])
        return q

    v = np.zeros(env.nS)
    count = 0
    while True:
        delta = 0
        for s in range(env.nS):
            q = action_value_function(s, v)
            best_action_value = np.max(q)
            delta = max(delta, np.abs(best_action_value - v[s]))
            v[s] = best_action_value  # v*(s)
        count += 1
        print('after ' + str(count) + ' iterations, the value function is: ')
        print(v)
        if delta < threshold:
            break
    print('optimal value function (v*(s)): ')
    print(v)
    # get policy
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        q = action_value_function(s, v)
        best_a_arr, policy_arr = get_max_index(q)
        policy[s] = policy_arr
    return policy, v


if __name__ == "__main__":
    env = gym.make("GridWorld_sv0")
    policy, v = value_iteration(env)
    print("Optimal Value function:")
    print(v)
    print("Optimal Policy:")
    print(policy)
    draw_policy(policy, title_name='Visualization of optimal policy')
