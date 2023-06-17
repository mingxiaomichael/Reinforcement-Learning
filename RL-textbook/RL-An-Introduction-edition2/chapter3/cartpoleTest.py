import gymnasium as gym
env = gym.make("CartPole-v0", render_mode="human")
state = env.reset()  # initialize the env

for t in range(500):
    env.render()
    print(state)
    action = env.action_space.sample()  # agent policy that uses the observation and info
    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Finished")
        state = env.reset()

env.close()
