import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GenericNetwork(nn.Module):
    def __init__(self, input_dims, p_fc_dims, v_fc_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.p_fc_dims = p_fc_dims
        self.v_fc_dims = v_fc_dims
        self.n_actions = n_actions
        self.p_fc1 = nn.Linear(self.input_dims, self.p_fc_dims[0])
        self.v_fc1 = nn.Linear(self.input_dims, self.v_fc_dims[0])
        self.p_fc2 = nn.Linear(self.p_fc_dims[0], self.p_fc_dims[1])
        self.v_fc2 = nn.Linear(self.v_fc_dims[0], self.v_fc_dims[1])
        self.mu = nn.Linear(self.p_fc_dims[-1], self.n_actions)
        self.log_sigma = nn.Linear(self.p_fc_dims[-1], self.n_actions)
        self.v = nn.Linear(self.v_fc_dims[-1], 1)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Actor
        x = F.relu(self.p_fc1(state))
        x = F.relu(self.p_fc2(x))
        mu = F.tanh(self.mu(x))
        log_sigma = self.log_sigma(x)
        # Critic
        x = F.relu(self.v_fc1(state))
        x = F.relu(self.v_fc2(x))
        v = self.v(x)
        return mu, log_sigma, v


class Agent(object):
    def __init__(self, input_dims, p_fc_dims, v_fc_dims, n_actions, lr_p, lr_v, gamma,
                 steps_to_update=32):
        self.n_actions = n_actions
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        self.log_prob = None

        self.ac = GenericNetwork(input_dims, p_fc_dims, v_fc_dims, self.n_actions)
        self.ac_target = GenericNetwork(input_dims, p_fc_dims, v_fc_dims, self.n_actions)
        self.mem_cntr = 0
        self.steps_to_update = steps_to_update

    def choose_action(self, state):
        state = T.tensor(state, dtype=T.float).to(self.ac.device)
        mu, log_sigma, _ = self.ac.forward(state)
        distribution = T.distributions.Normal(mu, log_sigma.exp())
        action = distribution.rsample()
        action.to(self.ac.device)
        self.log_prob = distribution.log_prob(action).to(self.ac.device)
        action = T.clamp(action, min=-1, max=1)
        return action.item()

    def update_target(self, steps_to_update):
        if self.mem_cntr % steps_to_update == 0:
            self.ac_target.load_state_dict(self.ac.state_dict())

    def learn(self, state, action, reward, next_state, is_done):
        state = T.from_numpy(state)
        next_state = T.from_numpy(next_state)
        _, _, q_eval = self.ac.forward(state)
        with T.no_grad():
            _, _, q_next = self.ac_target.forward(next_state)
        if is_done:
            q_next = 0.0
        q_target = reward + self.gamma * q_next

        # the optimizer of Actor network
        optimizer_p = optim.Adam(self.ac.parameters(), lr=self.lr_p)
        optimizer_p.zero_grad()
        loss_p = -1 * self.log_prob * ((q_eval - q_target).detach())
        loss_p.sum().backward(retain_graph=True)

        # the optimizer of Critic network
        optimizer_v = optim.Adam(self.ac.parameters(), lr=self.lr_v)
        optimizer_v.zero_grad()
        loss_v = 0.5 * T.mean((q_eval - q_target) ** 2)
        loss_v.backward()

        optimizer_p.step()
        optimizer_v.step()

        self.update_target(self.steps_to_update)
