import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.Tensor([1, 2, 3])
b = F.softmax(a, dim=0)
print(b)

action_dist = torch.distributions.Categorical(b)
print(action_dist)
action = action_dist.sample()
print(action)
print(action.item())

