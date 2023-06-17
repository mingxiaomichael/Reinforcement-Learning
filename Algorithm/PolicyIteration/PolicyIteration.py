import gym
import numpy as np
from PolicyPlot import draw_policy


def policy_eval(policy, env, n, discount_factor=1.0, threshold=0.00001):
    # get the index of maximum action value (q(s, a))
    # def get_max_index(action_values):
    #     indices = []
    #     policy_arr = np.zeros(len(action_values))
    #     action_max_value = np.max(action_values)
    #     for i in range(len(action_values)):
    #         action_value = action_values[i]
    #         if action_value == action_max_value:
    #             indices.append(i)
    #             policy_arr[i] = 1
    #     return indices, policy_arr
    count = 0
    V = np.zeros(env.nS)  # value function
    policy_mediate = np.ones([env.nS, env.nA]) / env.nA
    while True:
        value_delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * (reward + discount_factor * prob * V[next_state])
            value_delta = max(value_delta, np.abs(v - V[s]))
            V[s] = v
        count += 1
        print('value function:')
        print(V)
        print('iteration number of convergence of value function: ' + str(count))
        if value_delta < threshold:
            break
    return np.array(V)


def policy_improvement(v, policy, discount_factor=1.0):
    def get_max_index(action_values):
        indexes = []
        policy_arr = np.zeros(len(action_values))
        action_max_value = np.max(action_values)
        for i in range(len(action_values)):
            action_value = action_values[i]
            if action_value == action_max_value:
                indexes.append(i)
                policy_arr[i] = 1
        return indexes, policy_arr

    policy_stable = True
    for s in range(env.nS):
        chosen_a = np.argmax(policy[s])
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += (reward + discount_factor * prob * v[next_state])
        best_a_arr, policy_arr = get_max_index(action_values)
        if chosen_a not in best_a_arr:
            policy_stable = False
        policy[s] = policy_arr
    return policy_stable, policy


def policy_iteration(env):
    policy = np.ones([env.nS, env.nA]) / env.nA
    n = 0
    while True:
        v = policy_eval(policy, env, n)
        n += 1
        if n == 1:
            print('价值函数的初值为：')
            print(v)
        policy_stable, policy = policy_improvement(v, policy)
        if policy_stable:
            return policy, v


if __name__ == "__main__":
    env = gym.make("GridWorld_sv0")
    policy, v = policy_iteration(env)
    print("最优策略为:")
    print(policy)
    print("最终的价值函数:")
    print(v)
    draw_policy(policy, title_name='Visualization of optimal policy')
    # env = gym.make('FrozenLake-v0')
    # env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
    # print(env.action_space.n)
    # print(env.observation_space.n)
    # print(env.P)
    # env.reset()
    # env.render()
    # a = [2, 2, 1, 1, 1, 2]

    # for t in a:
    #     env.render()
    #     # action = env.action_space.sample()
    #     action = t
    #     new_state, reward, is_truncated, is_finished, prob = env.step(action)
    #     print(new_state, reward, is_truncated, is_finished)
