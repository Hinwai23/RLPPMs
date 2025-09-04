import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym

from ELR_env.csv_to_gym_ELR import ELREnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.policy import RainbowPolicy


class RainbowGreedyHelper:
    def __init__(self, model_path: str = 'rainbow_elr_best.pth') -> None:
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}

        # Environment to obtain valid actions and transitions
        self.env = ELREnv(all_transitions, activity2idx, use_true_end_reward=True)

        # Load Rainbow model (architecture must match training)
        state_shape = (29,)
        action_shape = len(self.activity2idx)

        def noisy_linear(x, y):
            return NoisyLinear(x, y, 0.1)

        device = 'cpu'
        net = Net(
            state_shape,
            action_shape,
            [256, 256],
            torch.nn.Mish,
            softmax=True,
            num_atoms=51,
            dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
            device=device,
        ).to(device)

        policy = RainbowPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters()),
            discount_factor=0.99,
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            estimation_step=1,
            target_update_freq=500,
            action_space=gym.spaces.Discrete(action_shape),
        )

        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict)
        policy.eval()
        self.policy = policy

        # Precompute support for expected value calculation
        self.num_atoms = 51
        self.v_min = -10.0
        self.v_max = 10.0
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

    def _q_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: shape (action_dim, num_atoms)
        returns: q_values: shape (action_dim,)
        """
        probs = torch.softmax(logits, dim=-1)
        q_values = torch.sum(probs * self.support.to(probs.device), dim=-1)
        return q_values

    def get_optimal_action(self, state_str: str) -> str | None:
        state_vec = encode_state(state_str)

        # Set current state for valid action mask
        self.env.current_state_str = state_str
        self.env.current_state_vec = state_vec
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None

        self.policy.model.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            # model output for Rainbow: logits with shape (batch, action_dim, num_atoms)
            logits, _ = self.policy.model(obs_tensor)
            logits = logits.squeeze(0)  # (action_dim, num_atoms)
            q_values = self._q_from_logits(logits)  # (action_dim,)

            masked_q_values = q_values.clone()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = masked_q_values + mask

            optimal_action_idx = torch.argmax(masked_q_values).item()

        return self.idx2activity[optimal_action_idx]


def simulate_case_acceptance(case_df: pd.DataFrame, prefix: int, helper: RainbowGreedyHelper) -> bool:
    """
    - if case rows < prefix: check if 'O_ACCEPTED' appears; if yes, accept, otherwise reject.
    - otherwise: if 'O_ACCEPTED' appears before prefix, accept; otherwise, from the prefix-th state, use the optimal valid action
      and the corresponding next state to scroll until 'END' or 'O_ACCEPTED' is selected.
    return a boolean value.
    """
    num_rows = len(case_df)
    actions_series = case_df['a'] if 'a' in case_df.columns else pd.Series(dtype=object)

    if num_rows < prefix:
        return ('O_ACCEPTED' in set(actions_series.values))

    pre_actions = actions_series.iloc[: max(prefix - 1, 0)]
    if 'O_ACCEPTED' in set(pre_actions.values):
        return True

    start_state = case_df.iloc[prefix - 1]['s']
    if start_state == 'END':
        return False

    current_state = start_state
    max_steps = 2000
    for _ in range(max_steps):
        optimal_action = helper.get_optimal_action(current_state)
        if optimal_action is None:
            break
        if optimal_action == 'O_ACCEPTED':
            return True

        next_tuple = helper.env.transitions.get(current_state, {}).get(optimal_action, (None, None))
        next_state = next_tuple[0]
        if next_state is None or next_state == 'END':
            break
        current_state = next_state

    return False


def evaluate_prefix_range(test_file_path: str, prefixes: list[int]) -> None:
    # a: sum of reward column in test file
    test_df = pd.read_csv(test_file_path)
    a_value = float(test_df['reward'].sum()) if 'reward' in test_df.columns else 0.0

    # prepare: group by case, pre-fetch amount of each case (take the first row)
    grouped = test_df.groupby('case')
    total_cases = int(test_df['case'].nunique())
    case_amount = {}
    for case_id, g in grouped:
        if 'amount' in g.columns and len(g) > 0:
            case_amount[case_id] = float(g.iloc[0]['amount'])
        else:
            case_amount[case_id] = 0.0

    # load Rainbow greedy helper (load once, reuse many times)
    helper = RainbowGreedyHelper('rainbow_elr_best.pth')

    for prefix in prefixes:
        accepted_cases = set()
        for case_id, g in grouped:
            g = g.reset_index(drop=True)
            accepted = simulate_case_acceptance(g, prefix, helper)
            if accepted:
                accepted_cases.add(case_id)

        # b: sum of amount of accepted cases * 0.15
        b_value = 0.15 * sum(case_amount.get(cid, 0.0) for cid in accepted_cases)
        avg_delta = (b_value - a_value) / total_cases if total_cases > 0 else 0.0
        print(f"prefix={prefix}, avg_delta={avg_delta}")


def main() -> None:
    test_file_path = 'preprocess/logs/80_20/MDP/event_log_rare_10000_cumulative_rewards_testing_20_mdp.csv'
    prefixes = list(range(1, 16))
    evaluate_prefix_range(test_file_path, prefixes)


if __name__ == '__main__':
    main()


