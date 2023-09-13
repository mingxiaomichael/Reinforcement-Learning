import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
    Value Network: Baseline
"""


class ValueNetwork(nn.Module):
    def __init__(self, lr_v, input_dims, fc1_dims, fc2_dims):
        super(ValueNetwork, self).__init__()
        self.lr = lr_v
        self.input_dims = input_dims
        self.input_rows = self.input_dims[0]
        self.input_columns = self.input_dims[1]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.conv1 = nn.Conv2d(self.input_rows, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2560, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_v)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = x.view(x.size(0) * x.size(1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.input_rows = self.input_dims[0]
        self.input_columns = self.input_dims[1]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(self.input_rows, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2560, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = x.view(x.size(0) * x.size(1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


class Agent():
    def __init__(self, gamma, lr, lr_v, input_dims, n_actions):
        self.gamma = gamma
        self.lr = lr
        self.lr_v = lr_v
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.PolicyNet = PolicyNetwork(self.lr, self.input_dims,
                                       fc1_dims=256, fc2_dims=64, n_actions=n_actions)
        self.ValueNet = ValueNetwork(self.lr_v, self.input_dims,
                                     fc1_dims=256, fc2_dims=64)

    def choose_action(self, state):  # Randomly sampling action according to policy
        # Input state: [-0.03664991 -0.01390656  0.04680819 -0.03493961]
        # Transfer (4,) list to (1, 4) Tensor
        state = T.from_numpy(state).float()
        probs = self.PolicyNet.forward(state)
        p = np.squeeze(probs.detach().numpy())
        # sampling action according to probability distribution
        action = np.random.choice(self.action_space, p=p)
        # log(prob)
        log_prob = T.log(probs.squeeze(0)[action])
        return action, log_prob
        # probs: tensor([[0.5784, 0.4216]], grad_fn= < SoftmaxBackward0 >)
        # probs.detach(): tensor([[0.5784, 0.4216]])
        # probs.squeeze(0): tensor([0.5784, 0.4216], grad_fn= < SqueezeBackward1 >)

    def discounted_future_reward(self, rewards):  # calculate Gt (return)
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt += (self.gamma ** pw) * r
                pw += 1
            discounted_rewards.append(Gt)
        return discounted_rewards

    def learn(self, state_list, rewards, log_probs):
        state_list = T.tensor(state_list).float()
        discounted_rewards = self.discounted_future_reward(rewards)
        discounted_rewards = T.tensor(discounted_rewards)
        policy_grads = []
        v = []
        for state, log_prob, Gt in zip(state_list, log_probs, discounted_rewards):
            baseline = self.ValueNet(state)
            advantage = Gt - baseline
            policy_grads.append(-1 * log_prob * advantage)
            v.append(baseline)
        v = T.tensor(v, requires_grad=True)
        # using stack sum can make parameters upgrading effectively
        self.PolicyNet.optimizer.zero_grad()
        policy_grad = T.stack(policy_grads).sum()
        policy_grad.backward()  # calculate gradient
        self.PolicyNet.optimizer.step()  # gradient ascent

        # updating Value Network
        self.ValueNet.optimizer.zero_grad()
        loss = self.ValueNet.loss(v, discounted_rewards).to(self.ValueNet.device)
        v.retain_grad()
        loss.backward()  # calculate gradient
        self.ValueNet.optimizer.step()  # gradient ascent
