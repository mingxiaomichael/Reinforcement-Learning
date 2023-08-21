import gym

env = gym.make('MountainCar-v0', render_mode="human")
print(env.action_space.n)
print(env.observation_space)
print(env.reset()[0])
state = env.reset()

total_steps = 0
total_reward = 0

# 最大交互次数
max_steps = 1000

# 与环境交互
for step in range(max_steps):
    # 渲染环境
    env.render()

    # action = env.action_space.sample()
    action = 2
    # 执行选择的动作，获取下一个状态、奖励和是否结束标志
    next_state, reward, is_terminated, _, _ = env.step(action)
    # 累计奖励
    total_reward += reward
    state = next_state
    print(step, action, state, reward, is_terminated)
    # 停止条件：达到目标位置或达到最大步数
    if is_terminated or step == max_steps - 1:
        print(step)
        print(f"Episode finished after {step + 1} steps. Total reward: {total_reward}")
        break

# 关闭环境渲染
env.close()


# import numpy as np
# import torch as T
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# a = T.tensor([1])
# b = T.tensor([1.0,2.0,3.0])
# print(a+b)
# c = b.mean()
# print(c)
# print(c + b)
