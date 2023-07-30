import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class SimpleDeepQNetwork(nn.Module):
    """
    This is a simple Deep Q Network with experience replay: Using neural network to train
    data/experience, then using Q-learning to compute TD error (Q_target - Q_evaluate),
    Q_evaluate is Q(s, a) from neural network, Q_target is the max value of Q(s', a'),
    where a' = {a1, a2, a3, a4}. Q(s', a') is from the same neural network as Q(s, a).

    The structure of Deep Q Network: 3 full connected layer

    Experience Replay: The agent can be trained until having enough experience data, the
    number of experience data is normally 10^4 ~ 10^5. Using experience replay can improve
    the performance of reinforcement learning.

    Input: state (len=8) --> DQN --> Output: Q value (len=4, 4 actions)
    """

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(SimpleDeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # full connected layer 1, (8, 256)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # full connected layer 2, (256, 256)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)  # full connected layer 3, (256, 4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # gradient descent, lr=0.003
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=0.005):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = SimpleDeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                         fc1_dims=256, fc2_dims=256)
        # Define variable of storing experience for experience replay
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, state, action, reward, state_, done):
        """
        Experience Replay
        Storing experience data, creating replay buffer.
        Replay Buffer: Experience array with length of max_mem_size=100000
                       state_memory, action_memory, reward_memory,
                       new_state_memory, terminal_memory
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        """
        NOTE:
        1. The number of stores equals "batch_size" means that DQN has enough data to learn.
        2. During each DQN learning, zeroing out the gradient of the network's parameters to
        avoid the accumulation of the gradient. ("self.Q_eval.optimizer.zero_grad()")
        3. During each DQN learning, the Agent will learn from experiences with length of
        "batch_size"
        4. Q_eval.forward(state_batch): batch_size(64) * action_number(8) Tensor matrix
           Q_eval.forward(state_batch)[batch_index, action_batch]: batch_size(64) Tensor vector
        """
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()  # zero out the gradients of the network's parameters
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        # End of this code behind, '[0]' means get the value Tensor of T.max(q_next, dim=1)
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()  # Compute the gradient of Loss with respect to the network parameters
        self.Q_eval.optimizer.step()  # gradient descent
