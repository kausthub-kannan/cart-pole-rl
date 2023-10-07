import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 128) # 4 because there are 4 parameters as the observation space
        self.actor = nn.Linear(128, 2) # 2 for the number of actions
        self.critic = nn.Linear(128, 1) # Critic is always 1
        self.saved_actions = []
        self.rewards = []
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values
