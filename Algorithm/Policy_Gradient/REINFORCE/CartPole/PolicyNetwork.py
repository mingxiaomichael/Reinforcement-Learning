import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
    Why do we use policy gradient?
    1. more suitable for random policy; 2. more suitable for continuous (or much more) actions.
    
    How to get policy gradient and update policy network?
    There is policy in state value function -> policy network -> policy gradient
    gradient ascent -> updating parameters of policy net
    Do gradient ascent on theta is to do gradient descent on -1 * theta.
    *Using REINFORCE algorithm to calculate Q(s,a)*
    
    Why do we use REINFORCE algorithm?
    It is a unbiased evaluation of Q(s,a), because the trajectory is based on policy.
"""


class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Agent():
    def __init__(self, gamma, lr, input_dims, n_actions):
        self.gamma = gamma
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.PolicyNet = PolicyNetwork(self.lr, self.input_dims,
                                       fc1_dims=256, fc2_dims=64, n_actions=n_actions)

    def choose_action(self, state):  # Randomly sampling action according to policy
        # Input state: [-0.03664991 -0.01390656  0.04680819 -0.03493961]
        # Transfer (4,) list to (1, 4) Tensor
        state = T.from_numpy(state).float().unsqueeze(0)
        # probs: policy (the probability of action), (1, 2) Tensor
        with T.no_grad():
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

    def update_policy(self, rewards, log_probs):
        discounted_rewards = self.discounted_future_reward(rewards)
        discounted_rewards = T.tensor(discounted_rewards)
        policy_grads = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_grads.append(-1 * log_prob * Gt)
        # using stack sum can make parameters upgrading effectively
        self.PolicyNet.optimizer.zero_grad()
        policy_grad = T.stack(policy_grads).sum()
        policy_grad.backward()  # calculate gradient
        self.PolicyNet.optimizer.step()  # gradient ascent
