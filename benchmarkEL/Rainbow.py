import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gymnasium as gym

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import RainbowPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import NoisyLinear
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter
import datetime

from EL_env.csv_to_gym_EL import ELEnv, activity2idx, train_transitions, all_transitions, MaskedEnvWrapper

def make_train_env():
    return MaskedEnvWrapper(
        ELEnv(train_transitions, activity2idx,
               use_true_end_reward=True, reward_scale=0.001)
    )

def make_eval_env():
    return MaskedEnvWrapper(
        ELEnv(all_transitions, activity2idx,
               use_true_end_reward=True, reward_scale=0.001)
    )


class RainbowNet(nn.Module):
    def __init__(self, state_shape, action_shape, num_atoms=51, hidden_size=256, noisy_std=0.5):
        super().__init__()
        self.num_atoms = num_atoms
        self.action_shape = action_shape
        
        self.feature = nn.Sequential(
            nn.Linear(state_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_size, num_atoms, noisy_std)
        )
        
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_size, action_shape * num_atoms, noisy_std)
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        features = self.feature(obs)
        
        value = self.value_stream(features)  # [batch, num_atoms]
        advantage = self.advantage_stream(features)  # [batch, action_shape * num_atoms]
        
        advantage = advantage.view(-1, self.action_shape, self.num_atoms)  # [batch, action, atoms]
        value = value.view(-1, 1, self.num_atoms)  # [batch, 1, atoms]
        
        advantage_mean = advantage.mean(dim=1, keepdim=True)  # [batch, 1, atoms]
        q_atoms = value + advantage - advantage_mean  # [batch, action, atoms]

        
        return q_atoms, state



def save_best_fn(policy):
    # Save policy network
    torch.save(policy.model.state_dict(), 'rainbow_el_best.pth')


if __name__ == '__main__':
    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = f"training_logs/rainbow_el_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    # Create vectorized environments
    train_envs = DummyVectorEnv([make_train_env for _ in range(4)])
    eval_envs = DummyVectorEnv([make_eval_env for _ in range(4)])
    # eval_env = make_eval_env()
    train_envs.seed(seed)
    eval_envs.seed(seed)

    # Extract shapes
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n

    device = 'cpu'

    # Build network and optimizer
    net = RainbowNet(
        state_shape=state_shape,
        action_shape=action_shape,
        num_atoms=51,
        hidden_size=256,
        noisy_std=0.5
    ).to(device)

    optim = Adam(net.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)

    # Configure Rainbow policy
    policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=0.99,
        num_atoms=51,
        v_min=-10.0, v_max=10.0,
        estimation_step=1,
        target_update_freq=500,
        action_space=train_envs.action_space[0],
    )
    # policy.set_eps(0.5)

    # Replay buffer with prioritized replay
    buffer = VectorReplayBuffer(
        total_size=50000,
        buffer_num=len(train_envs)
    )



    # Collectors
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, eval_envs)

    train_collector.reset()
    test_collector.reset()

    train_collector.collect(n_step=5000)

    # Trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=5000,
        step_per_collect=500,
        episode_per_test=20,
        batch_size=128,
        save_best_fn=save_best_fn,
        logger=logger,
        verbose=True,
    ).run()

    writer.close()

    print('Training completed')

