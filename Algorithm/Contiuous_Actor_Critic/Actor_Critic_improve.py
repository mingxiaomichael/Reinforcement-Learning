import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dims, p_fc_dims, v_fc_dims, n_actions):
        super(ActorCritic, self).__init__()
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
        state = T.tensor(state, dtype=T.float).to(self.device)
        # Actor
        x = F.relu(self.p_fc1(state))
        x = F.relu(self.p_fc2(x))
        mu = F.tanh(self.mu(x))
        log_sigma = self.log_sigma(x)
        distribution = T.distributions.Normal(mu, log_sigma.exp())
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action = T.tanh(action)

        # Critic
        x = F.relu(self.v_fc1(state))
        x = F.relu(self.v_fc2(x))
        v = self.v(x)
        return action, log_prob, v


class ActorCriticAgent(object):
    def __init__(self, input_dims, p_fc_dims, v_fc_dims, n_actions, lr_p, lr_v, gamma,
                 batch_size=32, max_mem_size=1000, steps_to_update=25):
        self.n_actions = n_actions
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.gamma = gamma
        # self.log_prob = None

        self.ac = ActorCritic(input_dims, p_fc_dims, v_fc_dims, self.n_actions)
        self.ac_target = ActorCritic(input_dims, p_fc_dims, v_fc_dims, self.n_actions)
        # self.actor = ActorCritic(input_dims, p_fc_dims, v_fc_dims, self.n_actions)

        self.batch_size = batch_size
        self.mem_cntr = 0
        self.mem_size = max_mem_size
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.log_prob_memory = T.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.is_terminated_memory = np.zeros(self.mem_size, dtype=np.bool8)

        self.steps_to_update = steps_to_update

    # def choose_action(self, state):
    #     mu, log_sigma, _ = self.ac.forward(state)
    #     print(self.mem_cntr)
    #     print(mu, log_sigma)
    #     distribution = T.distributions.Normal(mu, log_sigma.exp())
    #     action = distribution.sample()
    #     self.log_prob = distribution.log_prob(action).to(self.ac.device)
    #     print(self.log_prob)
    #     action = T.tanh(action)
    #     return action.item()

    def store_transition(self, state, log_prob, reward, next_action, is_terminated):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.log_prob_memory[index] = log_prob
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_action
        self.is_terminated_memory[index] = is_terminated
        self.mem_cntr += 1

    def update_target(self, steps_to_update):
        learn_steps = self.mem_cntr - self.batch_size
        if learn_steps >= 0 and learn_steps % steps_to_update == 0:
            self.ac_target.load_state_dict(self.ac.state_dict())

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # Training Batch
        state_batch = T.tensor(self.state_memory[batch]).to(self.ac.device)
        log_prob_batch = self.log_prob_memory[batch]
        # print("self.log_prob_memory[batch]: ", self.log_prob_memory[batch])
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.ac.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.ac.device)
        is_terminated_batch = T.tensor(self.is_terminated_memory[batch]).to(self.ac.device)
        # self.critic.load_state_dict(self.actor.state_dict())
        _, _, q_eval = self.ac.forward(state_batch)
        with T.no_grad():
            _, _, q_next = self.ac_target.forward(new_state_batch)
        # If is_terminated is True use reward, else update Q_target with discounted action values
        q_next[is_terminated_batch] = 0.0
        # End of this code behind, '[0]' means get the value Tensor of T.max(q_next, dim=1)
        q_target = reward_batch.reshape(32, 1) + self.gamma * q_next
        
        # the optimizer of Actor network
        optimizer_p = optim.Adam(self.ac.parameters(), lr=self.lr_p)
        optimizer_p.zero_grad()
        loss_p = -1 * log_prob_batch.reshape(32, 1) * ((q_eval - q_target).detach())
        loss_p.sum().backward(retain_graph=True)

        # the optimizer of Critic network
        optimizer_v = optim.Adam(self.ac.parameters(), lr=self.lr_v)
        optimizer_v.zero_grad()
        loss_v = 0.5 * T.mean((q_eval - q_target) ** 2)
        loss_v.sum().backward(retain_graph=True)

        optimizer_p.step()
        optimizer_v.step()

        self.update_target(self.steps_to_update)
