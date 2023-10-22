import numpy as np
import gym
import torch
from Algorithm.Contiuous_Actor_Critic.AC_replay_buffer.Actor_Critic_improve import ActorCriticAgent


if __name__ == '__main__':
    agent = ActorCriticAgent(input_dims=2, p_fc_dims=[64, 32], v_fc_dims=[64, 32], n_actions=1,
                             lr_p=0.001, lr_v=0.001, gamma=0.99)
    env = gym.make('MountainCarContinuous-v0', render_mode="human")
    scores = []
    steps = []
    num_episodes = 200
    for i in range(num_episodes):
        done = False
        score = 0
        state = env.reset()[0]
        step = 0
        while not done:
            action = np.array(agent.choose_action(state)).reshape((1,))
            next_state, reward, done, info, _ = env.step(action)
            agent.store_transition(state, reward, next_state, done)
            agent.learn()
            state = next_state
            score += reward
            step += 1
            if step % 100 == 0:
                print(step)
        scores.append(score)
        steps.append(step)
        print('episode: ', i, 'score: %.2f' % score)

    torch.save(agent.ac.state_dict(), '../ac.pth')
    torch.save(agent.ac_target.state_dict(), '../ac_target.pth')

    file_name = "../scores.txt"
    with open(file_name, "w") as file:
        for score in scores:
            file.write(f"{score}\n")

    file_name = "../steps.txt"
    with open(file_name, "w") as file:
        for step in steps:
            file.write(f"{step}\n")
