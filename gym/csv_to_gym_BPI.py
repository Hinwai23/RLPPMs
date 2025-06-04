import pandas as pd
import numpy as np
import gym
from gym import spaces
import copy
from collections import Counter


train_df = pd.read_csv('../preprocess/logs/80_20/MDP/BPI_2012_cumulative_rewards_training_80_mdp.csv')
test_df = pd.read_csv('../preprocess/logs/80_20/MDP/BPI_2012_cumulative_rewards_test_20_mdp.csv')

all_activities = sorted(train_df['a'].unique().tolist()) 
activity2idx = {act: i for i, act in enumerate(all_activities)}


raw_rewards = {}      # raw_rewards[s][a] = [r1, r2, r3, ...]
raw_next_states = {}  # raw_next_states[s][a] = [s'_1, s'_2, ...]

for _, row in train_df.iterrows():
    s  = row['s']
    a  = row['a']
    sp = row["s'"]
    r  = float(row['reward'])

    raw_rewards.setdefault(s, {}).setdefault(a, []).append(r)
    raw_next_states.setdefault(s, {}).setdefault(a, []).append(sp)


train_transitions = {}  # train_transitions[s][a] = (chosen_next_s, r_avg)

for s, adict in raw_next_states.items():
    for a, sp_list in adict.items():
        # sp_chosen = sp_list[0]
        
        sp_counter = Counter(sp_list)
        sp_chosen = sp_counter.most_common(1)[0][0]

        r_list = raw_rewards[s][a]
        r_avg = sum(r_list) / len(r_list)

        train_transitions.setdefault(s, {})[a] = (sp_chosen, r_avg)


test_transitions = {}  # test_transitions[s][a] = (next_s, r_true)

for _, row in test_df.iterrows():
    s      = row['s']
    a      = row['a']
    sp     = row["s'"]
    amount = float(row['amount'])


    r_true = float(row['reward'])

    test_transitions.setdefault(s, {})[a] = (sp, r_true)


all_transitions = copy.deepcopy(train_transitions)

for s, adict in test_transitions.items():
    for a, (sp, r_true) in adict.items():
        if s not in all_transitions:
            all_transitions[s] = {}
        if a not in all_transitions[s]:
            all_transitions[s][a] = (sp, r_true)



def encode_state(state_str):
    vec = np.zeros((29,), dtype=np.float32)
    if state_str == "START" or state_str == "END":
        return vec
    parts = state_str.split(',')
    activity = parts[0]
    call_after_offer = float(parts[1])
    call_for_missing = float(parts[2])
    num_offers   = float(parts[3])
    num_offers_back= float(parts[4])
    fix_incomplete = float(parts[5])

    idx = activity2idx[activity]
    vec[idx] = 1.0
    vec[24] = call_after_offer / 6.0
    vec[25] = call_for_missing / 6.0
    vec[26] = num_offers   / 7.0
    vec[27] = num_offers_back/ 3.0
    vec[28] = fix_incomplete / 2.0
    return vec

def scaled_reward(r_raw, is_end):
    # Step penalty
    step_penalty = 0
    if is_end:
        r_scale = min(r_raw / 2000.0, 1.0)
        return r_scale + step_penalty  
    else:
        r_mid = r_raw / 2000.0  
        return r_mid + step_penalty  
    



class BPIEnv(gym.Env):
    def __init__(self, transitions, activity2idx, use_true_end_reward=False):
        super().__init__()
        self.action2idx = {a: i for i, a in enumerate(activity2idx.keys())}
        self.idx2action = {i: a for a, i in self.action2idx.items()}
        self.transitions = transitions
        self.current_state_str = None
        self.current_state_vec = None

        self.use_true_end_reward = use_true_end_reward


        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(29,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action2idx))

    def reset(self):
        self.current_state_str = "START"
        self.current_state_vec = encode_state("START")
        return self.current_state_vec.copy()
    

    def step(self, action_idx):
        """
        input:action_idx(0..23)
        output:(next_obs, reward, done, info)
        
        - if (s, a) exists in transitions:
            next_s, r_stored = transitions[s][a]
            if use_true_end_reward=True and next_s=="END":
                reward = r_stored  
            else:
                reward = scaled_reward(r_stored, next_s=="END")
            done = (next_s == "END")
            return next_obs = encode_state(next_s), reward, done, {}
        - else:
            next_obs = encode_state("END")
            reward = -1.0  
            done = True
            info = {"unknown_transition": True}
        """
        s = self.current_state_str
        action_str = self.idx2action[action_idx]

        if s in self.transitions and action_str in self.transitions[s]:
            next_s, r_stored = self.transitions[s][action_str]
            if self.use_true_end_reward and next_s == "END":
                reward = r_stored
            else:
                reward = scaled_reward(r_stored, next_s == "END")

            done = (next_s == "END")
            next_vec = self.encode_state(next_s)
            self.current_state_str = next_s
            self.current_state_vec = next_vec.copy()
            return next_vec, reward, done, {}

        else:
            next_vec = self.encode_state("END")
            reward = -1.0
            done = True
            return next_vec, reward, done, {"unknown_transition": True}
        

    def render(self, mode='human'):
        print("STATE:", self.current_state_str)
    

"""
    def step(self, action_idx):
        action_str = self.idx2action[action_idx]

        if self.current_state_str not in self.transitions \
           or action_str not in self.transitions[self.current_state_str]:
            next_state_vec = self.current_state_vec.copy()
            return next_state_vec, -1.0, False, {"invalid": True}

        next_state_str, reward = self.transitions[self.current_state_str][action_str]
        done = (next_state_str == "END")
        if done:
            next_state_vec = encode_state("END")
        else:
            next_state_vec = encode_state(next_state_str)

        self.current_state_str = next_state_str
        self.current_state_vec = next_state_vec.copy()

        if next_state_str == "END":
            r = scaled_reward(reward, is_end=True)
        else:
            r = scaled_reward(reward, is_end=False)

        return next_state_vec, r, done, {}
"""

    