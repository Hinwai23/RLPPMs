import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym

from ELR_env.csv_to_gym_ELR import ELREnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import PPOPolicy


class PPOGreedyHelper:
    def __init__(self, model_path: str = 'ppo_elr_best.pth') -> None:
        self.model_path = model_path
        self.activity2idx = activity2idx

        # Environment to obtain valid actions and transitions
        self.env = ELREnv(all_transitions, activity2idx, use_true_end_reward=True)

        # Load PPO model (architecture must match training)
        state_shape = (29,)
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

        self.policy.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            actor_out = self.policy.actor(obs_tensor)
            logits = actor_out[0] if isinstance(actor_out, (tuple, list)) else actor_out
            logits = logits.squeeze(0)  # (action_dim,)

            # Greedy over valid actions
            masked_logits = logits.clone()
            mask = torch.ones_like(logits) * float('-inf')
            mask[valid_actions] = 0
            masked_logits = masked_logits + mask

            optimal_action_idx = torch.argmax(masked_logits).item()

        # Use environment's mapping for consistency
        return self.env.idx2action.get(optimal_action_idx)


def simulate_case_acceptance(case_df: pd.DataFrame, prefix: int, helper: PPOGreedyHelper) -> bool:
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

    # load PPO greedy helper (load once, reuse many times)
    helper = PPOGreedyHelper('ppo_elr_best.pth')

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
    prefixes = list(range(3, 16))
    evaluate_prefix_range(test_file_path, prefixes)


if __name__ == '__main__':
    main()


