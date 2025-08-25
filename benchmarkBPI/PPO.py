import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from BPI_env.csv_to_gym_BPI import BPIEnv, activity2idx, train_transitions, all_transitions, MaskedEnvWrapper


def make_train_env():
    return MaskedEnvWrapper(
        BPIEnv(train_transitions, activity2idx,
               use_true_end_reward=True, reward_scale=0.001)
    )

def make_eval_env():
    return MaskedEnvWrapper(
        BPIEnv(all_transitions, activity2idx,
               use_true_end_reward=True, reward_scale=0.001)
    )


def save_best_fn(policy):
    torch.save(policy.state_dict(), 'ppo_bpi_best.pth')




if __name__ == "__main__":

    log_dir = f"training_logs/ppo_bpi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n

    device = "cpu"

    # Networks
    net = Net(state_shape, hidden_sizes=[256,256],activation=torch.nn.Mish,device=device).to(device)
    actor = Actor(net, action_shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
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
        max_epoch=100,
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
