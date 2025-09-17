import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import gymnasium as gym
from collections import Counter

from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, NoisyLinear
from tianshou.policy import DQNPolicy, PPOPolicy, RainbowPolicy


class DQNGreedyHelper:
    def __init__(self, model_path: str = 'dqn_rtfm_best.pth') -> None:
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}
        self.env = RTFMEnv(all_transitions, activity2idx)
        state_shape = (len(self.activity2idx) + 2,)
        action_shape = len(self.activity2idx)
        net = Net(state_shape, action_shape, [256, 256], torch.nn.Mish, device='cpu').to('cpu')
        policy = DQNPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters()),
            action_space=gym.spaces.Discrete(action_shape),
            discount_factor=0.90,
            estimation_step=1,
            target_update_freq=500,
        )
        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict)
        policy.eval()
        self.policy = policy

    def get_optimal_action(self, state_str: str) -> str | None:
        state_vec = encode_state(state_str)
        self.env.current_state_str = state_str
        self.env.current_state_vec = state_vec
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None
        self.policy.model.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            q_values, _ = self.policy.model(obs_tensor)
            q_values = q_values.squeeze()
            masked_q_values = q_values.clone()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = masked_q_values + mask
            optimal_action_idx = torch.argmax(masked_q_values).item()
        return self.idx2activity[optimal_action_idx]


class PPOGreedyHelper:
    def __init__(self, model_path: str = 'ppo_rtfm_best.pth') -> None:
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.env = RTFMEnv(all_transitions, activity2idx)
        state_shape = (len(self.activity2idx) + 2,)
        action_shape = len(self.activity2idx)
        device = 'cpu'
        net = Net(state_shape, hidden_sizes=[256, 256], activation=torch.nn.Mish, device=device).to(device)
        actor = Actor(net, action_shape, device=device).to(device)
        critic = Critic(net, device=device).to(device)
        actor_critic = ActorCritic(actor, critic)
        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=torch.optim.Adam(actor_critic.parameters(), lr=3e-4),
            dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
            discount_factor=0.95,
            deterministic_eval=True,
            action_space=gym.spaces.Discrete(action_shape),
            action_scaling=False,
        )
        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict)
        policy.eval()
        self.policy = policy

    def get_optimal_action(self, state_str: str) -> str | None:
        state_vec = encode_state(state_str)
        self.env.current_state_str = state_str
        self.env.current_state_vec = state_vec
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None
        self.policy.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            actor_out, _ = self.policy.actor(obs_tensor)
            logits = actor_out if isinstance(actor_out, (tuple, list)) else actor_out
            logits = logits.squeeze(0)
            masked_logits = logits.clone()
            mask = torch.ones_like(logits) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = masked_logits + mask
            optimal_action_idx = torch.argmax(masked_logits).item()
        return self.env.idx2action.get(optimal_action_idx)


class RainbowGreedyHelper:
    def __init__(self, model_path: str = 'rainbow_rtfm_best.pth') -> None:
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}
        self.env = RTFMEnv(all_transitions, activity2idx)
        state_shape = (len(self.activity2idx) + 2,)
        action_shape = len(self.activity2idx)
        
        self.num_atoms = 51
        self.v_min = -10.0
        self.v_max = 10.0

        def noisy_linear(x, y):
            return NoisyLinear(x, y, 0.1)
        device = 'cpu'
        net = Net(
            state_shape,
            action_shape,
            [256, 256],
            torch.nn.Mish,
            softmax=True,
            num_atoms=self.num_atoms,
            dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
            device=device,
        ).to(device)
        policy = RainbowPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters()),
            discount_factor=0.99,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            estimation_step=1,
            target_update_freq=500,
            action_space=gym.spaces.Discrete(action_shape),
        )
        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict)
        policy.eval()
        self.policy = policy
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

    def _q_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        q_values = torch.sum(probs * self.support.to(probs.device), dim=-1)
        return q_values

    def get_optimal_action(self, state_str: str) -> str | None:
        state_vec = encode_state(state_str)
        self.env.current_state_str = state_str
        self.env.current_state_vec = state_vec
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None
        self.policy.model.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.policy.model(obs_tensor)

            support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
            probs = torch.softmax(logits, dim=-1)
            q_values = torch.sum(probs * support.unsqueeze(0).unsqueeze(0), dim=-1).squeeze(0)

            masked_q_values = q_values.clone()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = masked_q_values + mask
            optimal_action_idx = torch.argmax(masked_q_values).item()
        return self.idx2activity[optimal_action_idx]


def get_unique_states(test_file_path: str, prefix: int) -> list[str]:
    df = pd.read_csv(test_file_path)
    grouped = df.groupby('case')
    states = set()
    for _, g in grouped:
        if len(g) >= prefix:
            state = g.iloc[prefix - 1]['s']
            if state != 'END':
                states.add(state)
    return list(states)

def mode_next_state(state_str, action_str):
    pairs = all_transitions.get(state_str, {}).get(action_str, [])
    if not pairs:
        return None
    sp_counts = Counter(sp for sp, _ in pairs)
    sp_mode, _ = sp_counts.most_common(1)[0]
    return sp_mode

def generate_action_sequence(start_state: str, helper) -> list[str]:
    actions = []
    current_state = start_state
    max_steps = 200  # Prevent infinite loops
    for _ in range(max_steps):
        action = helper.get_optimal_action(current_state)
        if action is None:
            break
        
        actions.append(action)
        
        next_state = mode_next_state(current_state, action)
        
        if next_state is None or next_state == 'END':
            break
            
        current_state = next_state
        
    return actions

def main() -> None:
    test_file_path = 'preprocess/logs/80_20/MDP/Road_Traffic_Fine_Management_Process_cumulative_rewards_testing_20_mdp.csv'
    prefix = 3  # Example prefix

    print(f"Comparing trajectories for prefix {prefix}")

    unique_states = get_unique_states(test_file_path, prefix)
    print(f"Found {len(unique_states)} unique states at prefix {prefix}.")

    dqn_helper = DQNGreedyHelper()
    ppo_helper = PPOGreedyHelper()
    rainbow_helper = RainbowGreedyHelper()

    from itertools import zip_longest

    for state in unique_states:
        print("\n" + "="*85)
        print(f"Initial State: {state}")
        print("="*85)

        dqn_actions = generate_action_sequence(state, dqn_helper)
        ppo_actions = generate_action_sequence(state, ppo_helper)
        rainbow_actions = generate_action_sequence(state, rainbow_helper)

        print(f"{'Step':<5} | {'DQN Action':<25} | {'PPO Action':<25} | {'Rainbow Action':<25}")
        print("-" * 85)

        for i, (dqn_a, ppo_a, r_a) in enumerate(zip_longest(dqn_actions, ppo_actions, rainbow_actions, fillvalue='-')):
            if i >= 10:
                break
            print(f"{i+1:<5} | {dqn_a:<25} | {ppo_a:<25} | {r_a:<25}")
        

if __name__ == '__main__':
    main()
