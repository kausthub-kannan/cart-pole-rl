import gym 
import random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch.optim as optim
from model import ActorCritic
from itertools import count
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    # We calculate the losses and perform backprop in this function
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses =[]
    returns = []
    
    for r in model.rewards[::-1]:
        R = r + 0.99 * R # 0.99 is our gamma number
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    
    loss.backward()
    optimizer.step()
    
    del model.rewards[:]
    del model.saved_actions[:]

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

def train():
    running_reward = 10
    for i_episode in count(): # We need around this much episodes
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward
        finish_episode()
        if i_episode % 10 == 0: # We will print some things out
            print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}".format(
                i_episode, ep_reward, running_reward
            ))
        if running_reward > env.spec.reward_threshold:
            print("Solved, running reward is now {} and the last episode runs to {} time steps".format(
                    running_reward, t
            ))
            break

train()
