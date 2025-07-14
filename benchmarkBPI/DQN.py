import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import numpy as np


import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import copy



from tianshou.env import DummyVectorEnv
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import datetime

from BPI_env.csv_to_gym_BPI import BPIEnv, activity2idx, train_transitions, all_transitions, MaskedEnvWrapper




def make_train_env():
    return MaskedEnvWrapper(BPIEnv(train_transitions, activity2idx, use_true_end_reward=True, reward_scale=0.001))

def make_eval_env():
    return MaskedEnvWrapper(BPIEnv(all_transitions, activity2idx, use_true_end_reward=True, reward_scale=0.001))


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.fc1 = nn.Linear(state_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_shape)
        self.dropout = nn.Dropout(0.2)

    def forward(self, obs, state=None, info={}):
        device = next(self.parameters()).device
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        q = self.fc4(x)
        return q, state


def save_best_fn(policy):
    torch.save(policy.state_dict(), 'dqn_bpi_best.pth')


if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"training_logs/DQN_BPI_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer)
    
    train_envs = DummyVectorEnv([make_train_env for _ in range(4)])
    eval_env = make_eval_env()
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_envs.seed(seed)
    
    state_shape = train_envs.observation_space[0].shape  
    action_shape = train_envs.action_space[0].n  
    
    net = Net(state_shape, action_shape).to("cpu")
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    policy = DQNPolicy(
        model=net,
        optim=optim,
        action_space=train_envs.action_space[0],
        discount_factor=0.95,
        estimation_step=1,
        target_update_freq=500,
    )
    
    buffer = VectorReplayBuffer(total_size=50000, buffer_num=len(train_envs))
    
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    eval_collector = Collector(policy, eval_env)
    
    train_collector.reset()
    eval_collector.reset()
    
    train_collector.collect(n_step=5000)
    
    
    
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=eval_collector,
        max_epoch=50,            
        step_per_epoch=5000,     
        step_per_collect=500,    
        episode_per_test=20,     
        batch_size=128,          
        save_best_fn=save_best_fn,
        logger=logger,
        verbose=True
    ).run()

    writer.close()
    
    print("Finished training")

