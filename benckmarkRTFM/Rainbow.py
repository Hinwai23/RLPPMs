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

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.policy import RainbowPolicy


from torch.optim import Adam


import torch.optim as optim

from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, train_transitions, all_transitions, MaskedEnvWrapper

def make_train_env():
    return MaskedEnvWrapper(
        RTFMEnv(train_transitions, activity2idx)
    )

def make_eval_env():
    return MaskedEnvWrapper(
        RTFMEnv(all_transitions, activity2idx)
    )



def save_best_fn(policy):
    torch.save(policy.state_dict(), 'rainbow_rtfm_best.pth')


if __name__ == '__main__':


    log_dir = f"training_logs/rainbow_rtfm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    train_envs = DummyVectorEnv([make_train_env for _ in range(4)])
    eval_envs = DummyVectorEnv([make_eval_env for _ in range(2)])
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_envs.seed(seed)
    eval_envs.seed(seed)

    # Network and policy
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n

    def noisy_linear(x, y):
        return NoisyLinear(x, y, 0.1)

    device = "cpu"
    net = Net(
        state_shape,
        action_shape,
        [256,256],
        torch.nn.Mish,
        softmax=True,
        num_atoms=51,
        dueling_param=({"linear_layer": noisy_linear},
            {"linear_layer": noisy_linear}),
        device=device,
    )

    optim = Adam(net.parameters(), lr=1e-3)

    policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=0.99,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        estimation_step=1,
        target_update_freq=500,
        action_space=train_envs.action_space[0],
    )

    # Replay buffer and collectors
    buffer = PrioritizedVectorReplayBuffer(
        total_size=50000,
        buffer_num=len(train_envs),
        alpha=0.6,
        beta=0.4,
        weight_norm=True,
    )

    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, eval_envs, exploration_noise=True)

    train_collector.reset()
    test_collector.reset()

    train_collector.collect(n_step=5000)

    def train_fn(epoch, env_step):
        policy.set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.set_eps(0.)

    # Training
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=100,
        step_per_collect=1000,
        episode_per_test=20,
        batch_size=128,
        save_best_fn=save_best_fn,
        logger=logger,
        verbose=True,
        train_fn=train_fn,
        test_fn=test_fn,
    ).run()
    

    writer.close()

    print('Training completed')

