import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc_dims[0])
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])  # 1/8
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc_dims[0])

        self.fc2 = nn.Linear(self.fc_dims[0], self.fc_dims[1])
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])  # 1/8
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc_dims[1])

        self.mu = nn.Linear(self.fc_dims[1], self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -0.003, 0.003)
        T.nn.init.uniform_(self.mu.bias.data, -0.003, 0.003)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        mu = T.tanh(self.mu(x))
        return mu


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc_dims, n_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc_dims = fc_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc_dims[0])
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])  # 1/8
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc_dims[0])

        self.fc2 = nn.Linear(self.fc_dims[0], self.fc_dims[1])
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])  # 1/8
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc_dims[1])

        self.fc3 = nn.Linear(self.n_actions, self.fc_dims[1])
        self.q = nn.Linear(self.fc_dims[1], 1)
        T.nn.init.uniform_(self.q.weight.data, -0.003, 0.003)
        T.nn.init.uniform_(self.q.bias.data, -0.003, 0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        x1 = self.fc1(state)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x2 = F.relu(self.fc3(action))
        q = F.relu(T.add(x1, x2))
        q = self.q(q)
        return q


class ExplorationNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = 0.0
        self.decay_count = 0

    def ou_noise(self, sigma_decay=False):
        if sigma_decay:
            self.decay_count += self.dt
            self.sigma = self.sigma * np.exp(-self.decay_count)
        dW = np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        x = self.x_prev + (self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * dW)
        self.x_prev = x
        return x


class ReplayBuffer(object):
    def __init__(self, max_size, input_dims, n_actions):
        self.max_size = max_size
        self.state_memory = np.zeros((self.max_size, input_dims))
        self.action_memory = np.zeros((self.max_size, n_actions))
        self.reward_memory = np.zeros(self.max_size)
        self.next_state_memory = np.zeros((self.max_size, input_dims))
        self.terminal_memory = np.zeros(self.max_size, dtype=np.float32)
        self.counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        index = self.counter % self.max_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - done
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.max_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_state = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, next_state, terminal


class DDPG(object):
    def __init__(self, alpha, beta, gamma, input_dims, fc_dims, n_actions,
                 tau, device, max_size=50000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.actor = ActorNetwork(alpha, input_dims, fc_dims, n_actions)
        self.critic = CriticNetwork(beta, input_dims, fc_dims, n_actions)
        self.actor_target = ActorNetwork(alpha, input_dims, fc_dims, n_actions)
        self.critic_target = CriticNetwork(beta, input_dims, fc_dims, n_actions)
        self.replay_buffer = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.noise = ExplorationNoise(mu=np.zeros(n_actions))

    def soft_update(self, tau):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.critic_target.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.actor_target.load_state_dict(actor_state_dict)

    def choose_action(self, state):
        self.actor.eval()
        state = T.tensor(state, dtype=T.float).to(self.device)
        mu = self.actor.forward(state).to(self.device)
        mu = mu + T.tensor(self.noise.ou_noise(), dtype=T.float).to(self.device)
        self.actor.train()
        return mu.cpu().detach().numpy()

    def learn(self):
        if self.replay_buffer.counter < self.batch_size:
            return
        state, action, reward, next_state, is_done = self.replay_buffer.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.device)
        is_done = T.tensor(is_done).to(self.device)

        self.critic.eval()
        self.critic_target.eval()
        self.actor_target.eval()
        next_mu = self.actor_target.forward(next_state)
        q_next = self.critic_target.forward(next_state, next_mu)
        q_eval = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * q_next[j] * is_done[j])
        q_target = T.tensor(target).to(self.device)
        q_target = q_target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(q_target, q_eval)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.soft_update(self.tau)
