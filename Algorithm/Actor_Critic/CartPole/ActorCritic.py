import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, input_dims, output_dims, fc_dims):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, output_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, input_dims, fc_dims):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCritic(object):
    def __init__(self, input_dims, fc_dims, output_dims, actor_lr, critic_lr, gamma, device):
        self.gamma = gamma
        self.device = device
        # self.log_prob = None
        self.actor = PolicyNet(input_dims, output_dims, fc_dims).to(self.device)
        self.critic = ValueNet(input_dims, fc_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).to(self.device)
        with T.no_grad():
            probs = self.actor.forward(state)
        distribution = T.distributions.Categorical(probs)
        action = distribution.sample()
        # self.log_prob = distribution.log_prob(action).to(self.device)
        return action.item()

    def learn(self, transition_memory):
        state = T.tensor([t[0] for t in transition_memory], dtype=T.float).to(self.device)
        action = T.tensor([t[1] for t in transition_memory], dtype=T.float).to(self.device)
        reward = T.tensor([t[2] for t in transition_memory], dtype=T.float).to(self.device)
        next_state = T.tensor([t[3] for t in transition_memory], dtype=T.float).to(self.device)
        is_done = T.tensor([t[4] for t in transition_memory], dtype=T.float).to(self.device)
        # print("state: ", state)
        # print("action: ", action)
        # print("reward: ", reward)
        # print("next_state: ", next_state)
        # print("is_done: ", is_done)
        with T.no_grad():
            q_next = self.critic.forward(next_state)
        q_target = reward + self.gamma * q_next * (1 - is_done)
        td_error = self.critic.forward(state) - q_target
        log_prob = T.log(self.actor.forward(state).gather(1, action.unsqueeze(1).to(T.int64)))
        actor_loss = -1 * log_prob * td_error.detach()
        critic_loss = 0.5 * T.mean(td_error ** 2)
        print("critic loss: ", critic_loss)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.sum().backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

