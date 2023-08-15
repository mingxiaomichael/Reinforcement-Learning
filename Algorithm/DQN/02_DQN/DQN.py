import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
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
                 max_mem_size=100000, steps_to_update=25, tau=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.steps_to_update = steps_to_update
        self.tau = tau
        self.mem_cntr = 0

        # Evaluation Deep Q Network
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=64)
        # Fixed Deep Q Network
        self.Q_target = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                     fc1_dims=256, fc2_dims=64)

        # Experience Replay Space
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.is_terminated_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, state, action, reward, next_action, is_terminated):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_action
        self.is_terminated_memory[index] = is_terminated
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
        if self.mem_cntr < self.batch_size:
            return
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Training Batch
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        is_terminated_batch = T.tensor(self.is_terminated_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        # If is_terminated is True use reward, else update Q_target with discounted action values
        q_next[is_terminated_batch] = 0.0
        # End of this code behind, '[0]' means get the value Tensor of T.max(q_next, dim=1)
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()  # Compute the gradient of Loss with respect to the network parameters
        self.Q_eval.optimizer.step()  # gradient descent, update parameters of Q_eval
        self.update_Q_target(self.steps_to_update)


    def update_Q_target(self, steps_to_update):
        learn_steps = self.mem_cntr - self.batch_size
        if learn_steps >= 0 and learn_steps % steps_to_update == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())


"""
In the traditional Deep Q-Learning (DQN) algorithm, it is standard to use two networks: the Q-network (also known 
as the online network) and the target network. The reason for using two networks comes from the need to stabilize the 
training process.

We can use soft upgrade on Q_target network, in this way, the w (parameters) of Q_target can by upgraded
smoothly.
"""

# https://chat.openai.com/share/4653cfff-e1a2-4202-b86c-b92aed0198f8


