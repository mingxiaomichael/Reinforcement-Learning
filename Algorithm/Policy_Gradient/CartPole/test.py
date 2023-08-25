import gym

# 创建CartPole环境
env = gym.make('CartPole-v1', render_mode='human')

# 重置环境并获取初始状态
initial_state = env.reset()

done = False
total_reward = 0

a_n = env.action_space.n
print(a_n)
o_n = env.observation_space
print(o_n)
state = env.reset()[0]
print(state)

while not done:
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
    print(f"Action: {action}, next_state: {next_state}, Reward: {reward}, Total Reward: {total_reward}")

env.close()






