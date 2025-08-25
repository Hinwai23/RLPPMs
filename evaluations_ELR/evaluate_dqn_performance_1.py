import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import defaultdict

from ELR_env.csv_to_gym_ELR import ELREnv, activity2idx, encode_state, all_transitions
from tianshou.utils.net.common import Net
from tianshou.policy import DQNPolicy

class DQNEvaluator:
    def __init__(self, model_path='dqn_elr_best.pth'):
        """
        Initialize the DQN evaluator for ELR
        """
        self.model_path = model_path
        self.activity2idx = activity2idx
        self.idx2activity = {i: act for act, i in activity2idx.items()}
        
        # Load the trained DQN model
        self.policy = self._load_model()
        
        # Create environment for evaluation
        self.env = ELREnv(all_transitions, activity2idx, use_true_end_reward=True)
        
    def _load_model(self):
        """Load the trained DQN model - ELR version"""
        # Model architecture should match the training
        state_shape = (29,)  # From the environment
        action_shape = len(self.activity2idx)
        
        # Create the same network structure as in training
        net = Net(state_shape, action_shape,[256,256],torch.nn.Mish,device="cpu").to("cpu")
        
        # Create policy with minimal setup (no need for training parameters)
        policy = DQNPolicy(
            model=net,
            optim=torch.optim.Adam(net.parameters()),  
            action_space=gym.spaces.Discrete(action_shape),
            discount_factor=0.90,
            estimation_step=1, 
            target_update_freq=500,
        )
        
        # Load the saved model weights
        saved_state_dict = torch.load(self.model_path, map_location='cpu')
        
        # Load the filtered state dict
        policy.load_state_dict(saved_state_dict)
        
        # Set to evaluation mode
        policy.eval()
        
        return policy
    
    def get_optimal_action(self, state_str):
        """Get the optimal action according to DQN policy for a given state"""
        state_vec = encode_state(state_str)
        
        # Get valid actions for this state
        self.env.current_state_str = state_str
        self.env.current_state_vec = state_vec
        valid_actions = self.env.get_valid_actions()
        
        if not valid_actions:
            return None
        
        self.policy.model.eval()
            
        # Get Q-values from the model - use the network directly instead of policy
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            obs_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            
            # Use the network directly to get Q-values
            q_values, _ = self.policy.model(obs_tensor)
            q_values = q_values.squeeze()
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[valid_actions] = 0
            masked_q_values = masked_q_values + mask
            
            # Get the action with highest Q-value
            optimal_action_idx = torch.argmax(masked_q_values).item()
            
        return self.idx2activity[optimal_action_idx]
    
    def is_case_optimal(self, case_data, tolerance=0.2):
        """
        Check if a case follows the optimal policy
        case_data: DataFrame containing all transitions for one case
        """
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
            
        # Calculate the percentage of optimal steps
        optimal_ratio = optimal_steps / total_steps
        
        # Consider a case optimal if it's above the threshold
        return optimal_ratio >= (1.0 - tolerance)
    
    def calculate_offer_acceptance_rate(self, case_data):
        """
        Calculate offer acceptance rate for a case
        Check if the final reward is greater than 0
        """
        if len(case_data) == 0:
            return 0.0
            
        # Check if the final reward is greater than 0
        final_reward = case_data.iloc[-1].get('reward', 0) if 'reward' in case_data.columns else 0
        
        # If final reward > 0, consider it as offer accepted
        if final_reward > 0:
            return 1.0
        else:
            return 0.0
    
    def calculate_kpi(self, case_data):
        """
        Calculate KPI as the sum of all rewards in the case
        """
        if len(case_data) == 0:
            return 0.0
            
        # Sum all rewards in the case
        if 'reward' in case_data.columns:
            total_reward = case_data['reward'].sum()
        else:
            total_reward = 0.0
            
        return float(total_reward)
    
    def evaluate_performance(self, test_file_path):
        """
        Main evaluation function
        """
        print("Loading test data...")
        test_df = pd.read_csv(test_file_path)
        
        # Group by case
        cases = test_df.groupby('case')
        
        optimal_cases = []
        non_optimal_cases = []
        
        optimal_kpis = []
        non_optimal_kpis = []
        
        optimal_acceptance_rates = []
        non_optimal_acceptance_rates = []
        
        print(f"Evaluating {len(cases)} cases...")
        
        for i, (case_id, case_data) in enumerate(cases):
            if i % 100 == 0:
                print(f"Processed {i} cases...")
                
            # Sort by order of events
            case_data = case_data.reset_index(drop=True)
            
            # Check if case is optimal
            is_optimal = self.is_case_optimal(case_data)
            
            # Calculate metrics
            kpi = self.calculate_kpi(case_data)
            acceptance_rate = self.calculate_offer_acceptance_rate(case_data)
            
            if is_optimal:
                optimal_cases.append(case_id)
                optimal_kpis.append(kpi)
                optimal_acceptance_rates.append(acceptance_rate)
            else:
                non_optimal_cases.append(case_id)
                non_optimal_kpis.append(kpi)
                non_optimal_acceptance_rates.append(acceptance_rate)
        
        # Calculate statistics
        total_cases = len(optimal_cases) + len(non_optimal_cases)
        optimal_percentage = len(optimal_cases) / total_cases * 100 if total_cases > 0 else 0
        non_optimal_percentage = len(non_optimal_cases) / total_cases * 100 if total_cases > 0 else 0
        
        avg_optimal_kpi = np.mean(optimal_kpis) if optimal_kpis else 0
        avg_non_optimal_kpi = np.mean(non_optimal_kpis) if non_optimal_kpis else 0
        avg_all_kpi = np.mean(optimal_kpis + non_optimal_kpis) if (optimal_kpis + non_optimal_kpis) else 0
        
        avg_optimal_acceptance = np.mean(optimal_acceptance_rates) if optimal_acceptance_rates else 0
        avg_non_optimal_acceptance = np.mean(non_optimal_acceptance_rates) if non_optimal_acceptance_rates else 0
        avg_all_acceptance = np.mean(optimal_acceptance_rates + non_optimal_acceptance_rates) if (optimal_acceptance_rates + non_optimal_acceptance_rates) else 0
        
        return {
            'total_cases': total_cases,
            'optimal_cases': len(optimal_cases),
            'non_optimal_cases': len(non_optimal_cases),
            'optimal_percentage': optimal_percentage,
            'non_optimal_percentage': non_optimal_percentage,
            'avg_optimal_kpi': avg_optimal_kpi,
            'avg_non_optimal_kpi': avg_non_optimal_kpi,
            'avg_all_kpi': avg_all_kpi,
            'avg_optimal_acceptance': avg_optimal_acceptance * 100,
            'avg_non_optimal_acceptance': avg_non_optimal_acceptance * 100,
            'avg_all_acceptance': avg_all_acceptance * 100
        }
    
    def print_results(self, results):
        """
        Print results in the format shown in the table
        """
        print("=" * 80)
        print(f"{'traces':<15} {'trace #':<15} {'avg KPI':<10} {'Offer accepted':<20}")
        print("=" * 80)
        
        # All cases
        print(f"{'ALL':<15} {results['total_cases']:<15} {results['avg_all_kpi']:<10.1f} {results['avg_all_acceptance']:<20.1f}%")
        
        # Optimal cases  
        optimal_text = f"{results['optimal_cases']} ({results['optimal_percentage']:.1f}%)"
        print(f"{'OPTIMAL P.':<15} {optimal_text:<15} {results['avg_optimal_kpi']:<10.1f} {results['avg_optimal_acceptance']:<20.1f}%")
        
        # Non-optimal cases
        non_optimal_text = f"{results['non_optimal_cases']} ({results['non_optimal_percentage']:.1f}%)"
        print(f"{'NON-OPTIMAL P.':<15} {non_optimal_text:<15} {results['avg_non_optimal_kpi']:<10.1f} {results['avg_non_optimal_acceptance']:<20.1f}%")
        
        print("=" * 80)


def main():
    """
    Main function to run the evaluation for ELR
    """
    print("Initializing DQN evaluator (ELR)...")
    
    # Initialize evaluator with the saved model
    evaluator = DQNEvaluator('dqn_elr_best.pth')
    
    # Run evaluation on test data
    test_file_path = 'preprocess/logs/80_20/MDP/event_log_rare_10000_cumulative_rewards_testing_20_mdp.csv'
    
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
