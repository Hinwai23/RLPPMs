import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch import nn

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.discrete import Actor, Critic



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

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256,256,256,256)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.model = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, obs, state=None, info={}):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        return self.model(obs), state




if __name__ == "__main__":
    # Reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    log_dir = f"training_logs/ppo_el_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    # Vectorized environments
    train_envs = DummyVectorEnv([make_train_env for _ in range(4)])
    # eval_env = make_eval_env()
    eval_envs = DummyVectorEnv([make_eval_env for _ in range(4)])
    train_envs.seed(seed)
    eval_envs.seed(seed)

    # State and action dimensions
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n

    device = "cpu"


    net = Net(state_shape[0], hidden_dims=(256,256,256,256)).to(device)

    actor = Actor(net, action_shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)

    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=1e-4,
        eps=1e-5,
    )

    # PPO policy
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=0.95,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.05,
        reward_normalization=False,
        action_space=train_envs.action_space[0],
        action_scaling=False,
    )

    # On-policy collectors
    buffer = VectorReplayBuffer(total_size=50000, buffer_num=len(train_envs))
    train_collector = Collector(policy, train_envs, buffer)
    eval_collector = Collector(policy, eval_envs)

    train_collector.reset()
    eval_collector.reset()
    
    train_collector.collect(n_step=5000)

    # Training loop
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=eval_collector,
        max_epoch=50,
        step_per_epoch=10000,
        step_per_collect=8192,
        repeat_per_collect=20,
        episode_per_test=20,
        batch_size=128,
        logger=logger,
        verbose=True,
    ).run()

    writer.close()

    # Save best policy
    torch.save(policy.state_dict(), 'ppo_el_best.pth')
    
    print("Training completed!")
