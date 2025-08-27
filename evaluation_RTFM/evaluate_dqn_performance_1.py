import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym

from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy


class DQNEvaluator:
    def __init__(self, model_path='dqn_rtfm_best.pth'):
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}
        self.policy = self._load_model()
        self.env = RTFMEnv(all_transitions, activity2idx)

    def _load_model(self):
        state_shape = (len(self.activity2idx) + 2,)
        action_shape = len(self.activity2idx)
        net = Net(state_shape, action_shape, [256, 256], torch.nn.Mish, device="cpu").to("cpu")
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
            q_values, _ = self.policy.model(obs_tensor)
            q_values = q_values.squeeze()
            masked_q_values = q_values.clone()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = masked_q_values + mask
            optimal_action_idx = torch.argmax(masked_q_values).item()
        return self.idx2activity[optimal_action_idx]

    def calculate_full_payment(self, case_data):
        if len(case_data) == 0:
            return 0.0
        # full Payment: case contains at least one action 'Payment'
        has_payment = (case_data['a'] == 'Payment').any() if 'a' in case_data.columns else False
        return 1.0 if has_payment else 0.0

    def calculate_kpi(self, case_data):
        if len(case_data) == 0:
            return 0.0
        return float(case_data['reward'].sum()) if 'reward' in case_data.columns else 0.0

    def is_case_optimal(self, case_data, tolerance=0.2):
        total_steps = 0
        optimal_steps = 0
        for _, row in case_data.iterrows():
            state = row['s']
            actual_action = row['a']
            if state == 'END':
                continue
            total_steps += 1
            optimal_action = self.get_optimal_action(state)
            if optimal_action is None:
                continue
            if actual_action == optimal_action:
                optimal_steps += 1
        if total_steps == 0:
            return True
        return (optimal_steps / total_steps) >= (1.0 - tolerance)

    def evaluate_performance(self, test_file_path):
        print("Loading test data...")
        test_df = pd.read_csv(test_file_path)
        cases = test_df.groupby('case')
        optimal_cases, non_optimal_cases = [], []
        optimal_kpis, non_optimal_kpis = [], []
        optimal_fullpay, non_optimal_fullpay = [], []
        print(f"Evaluating {len(cases)} cases...")
        for i, (case_id, case_data) in enumerate(cases):
            if i % 100 == 0:
                print(f"Processed {i} cases...")
            case_data = case_data.reset_index(drop=True)
            is_opt = self.is_case_optimal(case_data)
            kpi = self.calculate_kpi(case_data)
            acc = self.calculate_full_payment(case_data)
            if is_opt:
                optimal_cases.append(case_id)
                optimal_kpis.append(kpi)
                optimal_fullpay.append(acc)
            else:
                non_optimal_cases.append(case_id)
                non_optimal_kpis.append(kpi)
                non_optimal_fullpay.append(acc)
        total_cases = len(optimal_cases) + len(non_optimal_cases)
        return {
            'total_cases': total_cases,
            'optimal_cases': len(optimal_cases),
            'non_optimal_cases': len(non_optimal_cases),
            'optimal_percentage': (len(optimal_cases) / total_cases * 100) if total_cases > 0 else 0,
            'non_optimal_percentage': (len(non_optimal_cases) / total_cases * 100) if total_cases > 0 else 0,
            'avg_optimal_kpi': np.mean(optimal_kpis) if optimal_kpis else 0,
            'avg_non_optimal_kpi': np.mean(non_optimal_kpis) if non_optimal_kpis else 0,
            'avg_all_kpi': np.mean((optimal_kpis + non_optimal_kpis)) if (optimal_kpis + non_optimal_kpis) else 0,
            'avg_optimal_full_payment': (np.mean(optimal_fullpay) * 100) if optimal_fullpay else 0,
            'avg_non_optimal_full_payment': (np.mean(non_optimal_fullpay) * 100) if non_optimal_fullpay else 0,
            'avg_all_full_payment': (np.mean(optimal_fullpay + non_optimal_fullpay) * 100) if (optimal_fullpay + non_optimal_fullpay) else 0,
        }

    def print_results(self, results):
        print("=" * 80)
        print("DQN EVALUATION RESULTS (RTFM)")
        print("=" * 80)
        print(f"{'traces':<15} {'trace #':<15} {'avg KPI':<10} {'full Payment':<20}")
        print("=" * 80)
        print(f"{'ALL':<15} {results['total_cases']:<15} {results['avg_all_kpi']:<10.1f} {results['avg_all_full_payment']:<20.1f}%")
        optimal_text = f"{results['optimal_cases']} ({results['optimal_percentage']:.1f}%)"
        print(f"{'OPTIMAL P.':<15} {optimal_text:<15} {results['avg_optimal_kpi']:<10.1f} {results['avg_optimal_full_payment']:<20.1f}%")
        non_optimal_text = f"{results['non_optimal_cases']} ({results['non_optimal_percentage']:.1f}%)"
        print(f"{'NON-OPTIMAL P.':<15} {non_optimal_text:<15} {results['avg_non_optimal_kpi']:<10.1f} {results['avg_non_optimal_full_payment']:<20.1f}%")


def main():
    print("Initializing DQN evaluator (RTFM)...")
    evaluator = DQNEvaluator('dqn_rtfm_best.pth')
    test_file_path = 'preprocess/logs/80_20/MDP/Road_Traffic_Fine_Management_Process_cumulative_rewards_testing_20_mdp.csv'
    try:
        results = evaluator.evaluate_performance(test_file_path)
        evaluator.print_results(results)
        return results
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


