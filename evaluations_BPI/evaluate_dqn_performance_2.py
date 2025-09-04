import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym

from BPI_env.csv_to_gym_BPI import BPIEnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy


class DQNGreedyHelper:
    def __init__(self, model_path: str = 'dqn_bpi_best.pth') -> None:
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}

        # Environment to obtain valid actions and transitions
        self.env = BPIEnv(all_transitions, activity2idx, use_true_end_reward=True)

        # Load DQN model (architecture must match training)
        state_shape = (29,)
        action_shape = len(self.activity2idx)
        net = Net(state_shape, action_shape, [256, 256], torch.nn.Mish, device='cpu').to('cpu')
        policy = DQNPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters()),
            action_space=gym.spaces.Discrete(action_shape),
            discount_factor=0.95,
            estimation_step=1,
            target_update_freq=500,
        )
        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        policy.load_state_dict(saved_state_dict)
        policy.eval()
        self.policy = policy

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
            q_values, _ = self.policy.model(obs_tensor)
            q_values = q_values.squeeze()

            masked_q_values = q_values.clone()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = masked_q_values + mask

            optimal_action_idx = torch.argmax(masked_q_values).item()

        return self.idx2activity[optimal_action_idx]


def simulate_case_acceptance(case_df: pd.DataFrame, prefix: int, helper: DQNGreedyHelper) -> bool:
    """
    determine if the case is accepted:
    - if case rows < prefix: check if 'O_ACCEPTED' appears; if yes, accept, otherwise reject.
    - otherwise: if 'O_ACCEPTED' appears before prefix, accept; otherwise, from the prefix-th state, use the optimal valid action
      and the corresponding next state to scroll until 'END' or 'O_ACCEPTED' is selected.
    return a boolean value.
    """
    num_rows = len(case_df)
    actions_series = case_df['a'] if 'a' in case_df.columns else pd.Series(dtype=object)

    if num_rows < prefix:
        return ('O_ACCEPTED' in set(actions_series.values))

    # 1-based: prefix-th state before -> prefix-1 rows
    pre_actions = actions_series.iloc[: max(prefix - 1, 0)]
    if 'O_ACCEPTED' in set(pre_actions.values):
        return True

    start_state = case_df.iloc[prefix - 1]['s']
    if start_state == 'END':
        return False

    current_state = start_state
    max_steps = 2000  # prevent infinite loop
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
    # a： total reward of all cases
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

    # load DQN greedy helper (load once, reuse many times)
    helper = DQNGreedyHelper('dqn_bpi_best.pth')

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
    test_file_path = 'preprocess/logs/80_20/MDP/BPI_2012_cumulative_rewards_testing_20_mdp.csv'
    prefixes = list(range(1, 16))
    evaluate_prefix_range(test_file_path, prefixes)


if __name__ == '__main__':
    main()


