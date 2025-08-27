import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import Counter

from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net
from tianshou.policy import RainbowPolicy
from tianshou.utils.net.discrete import NoisyLinear

torch.use_deterministic_algorithms(True)


class RainbowEvaluatorPrefix:
    def __init__(self, model_path='rainbow_rtfm_best.pth'):
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}
        self.num_atoms = 51
        self.v_min = -10.0
        self.v_max = 10.0
        self.policy = self._load_model()
        self.env = RTFMEnv(all_transitions, activity2idx)

    def _load_model(self):
        state_shape = (len(self.activity2idx) + 2,)
        action_shape = len(self.activity2idx)

        def noisy_linear(x, y):
            return NoisyLinear(x, y, 0.1)

        net = Net(
            state_shape,
            action_shape,
            [256, 256],
            torch.nn.Mish,
            softmax=True,
            num_atoms=self.num_atoms,
            dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
            device="cpu",
        )

        policy = RainbowPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters(), lr=1e-3),
            action_space=gym.spaces.Discrete(action_shape),
            discount_factor=0.99,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            estimation_step=1,
            target_update_freq=500,
        )

        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict)
        policy.eval()
        return policy

    def get_optimal_action(self, state_str):
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
    evaluator = RainbowEvaluatorPrefix('rainbow_rtfm_best.pth')
    test_file_path = 'preprocess/logs/80_20/MDP/Road_Traffic_Fine_Management_Process_cumulative_rewards_testing_20_mdp.csv'
    a, results = evaluator.evaluate_prefix(test_file_path, prefix_values=(1, 2, 3, 4, 5))
    print(f"Baseline reward sum a: {a:.2f}")
    for p in (1, 2, 3, 4, 5):
        print(f"prefix={p}: avg_delta={(results[p]):.4f}")


if __name__ == "__main__":
    main()


