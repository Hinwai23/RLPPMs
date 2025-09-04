import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from typing import Any, Tuple
from tianshou.utils.net.discrete import Actor as _TSActor
from tianshou.utils.net.discrete import Critic as _TSCritic

import gymnasium as gym
from gymnasium import spaces
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, train_transitions, all_transitions, ActionMaskObsWrapper


def make_train_env():
    return ActionMaskObsWrapper(
        RTFMEnv(train_transitions, activity2idx)
    )

def make_eval_env():
    return ActionMaskObsWrapper(
        RTFMEnv(all_transitions, activity2idx)
    )



def save_best_fn(policy):
    torch.save(policy.state_dict(), 'ppo_rtfm_best.pth')


class MaskedActor(_TSActor):

    def forward(self, obs: Any, state: Any = None, info: dict | None = None) -> Tuple[torch.Tensor, Any]:
        # Tianshou's collector passes a Batch object to the policy.
        # We need to unpack the actual observation and mask from this Batch object.
        raw_obs = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = super().forward(raw_obs, state, info)
        
        # Apply the action mask if it's available in the Batch object.
        if hasattr(obs, "mask"):
            mask = torch.as_tensor(obs.mask, device=logits.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(logits.shape[0], -1)
            elif mask.ndim == 2 and mask.shape[0] != logits.shape[0]:
                mask = mask.expand_as(logits)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            
        return logits, hidden
    
class MaskedCritic(_TSCritic):
    def forward(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        raw_obs = obs.obs if hasattr(obs, "obs") else obs
        return super().forward(raw_obs, **kwargs)


if __name__ == "__main__":

    log_dir = f"training_logs/ppo_rtfm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    logger = TensorboardLogger(writer)

    train_envs = DummyVectorEnv([make_train_env for _ in range(4)])
    eval_envs = DummyVectorEnv([make_eval_env for _ in range(2)])
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_envs.seed(seed)
    eval_envs.seed(seed)

    # Shapes
    state_shape = train_envs.observation_space[0]
    if isinstance(state_shape, spaces.Dict):
        state_shape = state_shape["obs"].shape
    else:
        state_shape = state_shape.shape
        
    action_shape = train_envs.action_space[0].n

    device = "cpu"

    # Networks
    net = Net(state_shape, hidden_sizes=[256,256],activation=torch.nn.Mish,device=device).to(device)
    actor = MaskedActor(net, action_shape, device=device).to(device)
    critic = MaskedCritic(net, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)

    optim = torch.optim.Adam(
        actor_critic.parameters(),
        lr=3e-4,
    )

    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # Policy
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
        discount_factor=0.95,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0,
        gae_lambda=0.95,
        advantage_normalization=False,
        reward_normalization=False,
        dual_clip=None,
        value_clip=False,
        action_space=train_envs.action_space[0],
        recompute_advantage=False,
        deterministic_eval=True,
        action_scaling=False,
    )

    # Collectors
    buffer = VectorReplayBuffer(total_size=50000, buffer_num=len(train_envs))
    train_collector = Collector(policy, train_envs, buffer)
    eval_collector = Collector(policy, eval_envs)

    # Trainer
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=eval_collector,
        max_epoch=50,
        step_per_epoch=100,
        step_per_collect=10000,
        repeat_per_collect=10,
        episode_per_test=100,
        batch_size=128,
        logger=logger,
        save_best_fn=save_best_fn,
        verbose=True,
    ).run()

    writer.close()
    
    print("Training completed!")
