import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import Counter

from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, encode_state, all_transitions
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic


class PPOEvaluatorPrefix:
    def __init__(self, model_path='ppo_rtfm_best.pth'):
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}
        self.policy = self._load_model()
        self.env = RTFMEnv(all_transitions, activity2idx)

    def _load_model(self):
        state_shape = (len(self.activity2idx) + 2,)
        action_shape = len(self.activity2idx)
        device = "cpu"
        base = Net(state_shape, hidden_sizes=[256, 256], activation=torch.nn.Mish, device=device).to(device)
        actor = Actor(base, action_shape, device=device).to(device)
        critic = Critic(base, device=device).to(device)
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
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
            action_space=gym.spaces.Discrete(action_shape),
            recompute_advantage=False,
            deterministic_eval=True,
            action_scaling=False,
        )
        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict, strict=False)
        policy.eval()
        return policy

    def get_optimal_action(self, state_str):
        state_vec = encode_state(state_str)
        self.env.current_state_str = state_str
        self.env.current_state_vec = state_vec
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None
        self.policy.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.policy.actor(obs_tensor)
            logits = logits.squeeze()
            masked_logits = logits.clone()
            mask = torch.ones_like(logits) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = masked_logits + mask
            optimal_action_idx = torch.argmax(masked_logits).item()
        return self.idx2activity[optimal_action_idx]

    def mode_next_state_and_reward(self, state_str, action_str):
        pairs = all_transitions.get(state_str, {}).get(action_str, [])
        if not pairs:
            return None, 0.0
        sp_counts = Counter(sp for sp, _ in pairs)
        sp_mode, _ = sp_counts.most_common(1)[0]
        rewards = [r for sp, r in pairs if sp == sp_mode]
        r_avg = float(np.mean(rewards)) if rewards else 0.0
        return sp_mode, r_avg

    def simulate_case_from_prefix(self, case_df, prefix):
        if len(case_df) < prefix:
            return 0.0
        s = case_df.iloc[prefix - 1]['s']
        total = 0.0
        steps = 0
        max_steps = len(case_df) + 100
        while s != 'END' and steps < max_steps:
            a = self.get_optimal_action(s)
            if a is None:
                break
            sp, r = self.mode_next_state_and_reward(s, a)
            if sp is None:
                break
            total += r
            s = sp
            steps += 1
        return total

    def evaluate_prefix(self, test_file_path, prefix_values=(1, 2, 3)):
        test_df = pd.read_csv(test_file_path)
        a = float(test_df['reward'].sum()) if 'reward' in test_df.columns else 0.0
        cases = test_df.groupby('case')
        num_cases = len(cases)
        results = {}
        for prefix in prefix_values:
            b = 0.0
            for _, case_df in cases:
                case_df = case_df.reset_index(drop=True)
                b += self.simulate_case_from_prefix(case_df, prefix)
            avg_delta = (b - a) / num_cases if num_cases > 0 else 0.0
            results[prefix] = avg_delta
        return a, results


def main():
    evaluator = PPOEvaluatorPrefix('ppo_rtfm_best.pth')
    test_file_path = 'preprocess/logs/80_20/MDP/Road_Traffic_Fine_Management_Process_cumulative_rewards_testing_20_mdp.csv'
    a, results = evaluator.evaluate_prefix(test_file_path, prefix_values=(1, 2, 3, 4, 5))
    print(f"Baseline reward sum a: {a:.2f}")
    for p in (1, 2, 3, 4, 5):
        print(f"prefix={p}: avg_delta={(results[p]):.4f}")


if __name__ == "__main__":
    main()


