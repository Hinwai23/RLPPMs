import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import torch
import numpy as np
import copy
from typing import Any, Dict, Optional, Union

import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.env import DummyVectorEnv
from tianshou.data import VectorReplayBuffer, Collector, Batch
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net
from tianshou.trainer import OffpolicyTrainer

import torch.optim as optim
from ELR_env.csv_to_gym_ELR import ELREnv, activity2idx, train_transitions, all_transitions, MaskedEnvWrapper


def make_train_env():
    return MaskedEnvWrapper(
        ELREnv(train_transitions, activity2idx, use_true_end_reward=True, reward_scale=0.001)
    )


def make_eval_env():
    return MaskedEnvWrapper(
        ELREnv(all_transitions, activity2idx, use_true_end_reward=True, reward_scale=0.001)
    )



def save_best_fn(policy):
    torch.save(policy.state_dict(), 'dqn_elr_best.pth')




if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"training_logs/DQN_ELR_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    logger = TensorboardLogger(writer)
    
    train_envs = DummyVectorEnv([make_train_env for _ in range(4)])
    eval_env = DummyVectorEnv([make_eval_env for _ in range(2)])

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_envs.seed(seed)
    eval_env.seed(seed)

    # Network and policy
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n
    net = Net(state_shape, action_shape,[256,256],torch.nn.Mish,device="cpu").to("cpu")
    optim_ = optim.Adam(net.parameters(), lr=1e-4)
    policy = DQNPolicy(
        model=net,
        optim=optim_,
        action_space=train_envs.action_space[0],
        discount_factor=0.90,
        estimation_step=1,
        target_update_freq=500,
    )

    # Data collectors
    buffer = VectorReplayBuffer(total_size=100000, buffer_num=len(train_envs))
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    eval_collector = Collector(policy, eval_env, exploration_noise=True)
    train_collector.reset()
    eval_collector.reset()
    train_collector.collect(n_step=5000)

    def train_fn(epoch, env_step):
        policy.set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.set_eps(0.)

    # Training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=eval_collector,
        max_epoch=50,
        step_per_epoch=100,
        step_per_collect=1000,
        episode_per_test=100,
        batch_size=128,
        save_best_fn=save_best_fn,
        logger=logger,
        verbose=True,
        train_fn=train_fn,
        test_fn=test_fn,
    ).run()

    writer.close()
    
    print("Finished training")