import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v2", render_mode="human")
model = DQN.load("dqn_Net256_lunar_2500K", env=env)
eposides = 10
show_steps = 300
for i in range(eposides):
    observation = env.reset()[0]
    done = False
    j = 0
    rewards = 0
    while (not done) and (j <= show_steps):
        env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation_, reward, done, info, _ = env.step(action)
        env.render()
        observation = observation_
        rewards += reward
        j += 1
    print("episode: ", i, " score: ",rewards)
env.close()
