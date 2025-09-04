import pandas as pd
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import copy
from collections import Counter
from gymnasium import Wrapper



train_df = pd.read_csv('preprocess/logs/80_20/MDP/event_log_rare_10000_cumulative_rewards_training_80_mdp.csv')
test_df = pd.read_csv('preprocess/logs/80_20/MDP/event_log_rare_10000_cumulative_rewards_testing_20_mdp.csv')

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
    vec[24] = call_after_offer / 25.0
    vec[25] = call_for_missing / 3.0
    vec[26] = num_offers   / 30.0
    vec[27] = num_offers_back/ 4.0
    vec[28] = fix_incomplete / 1.0
    return vec

def scaled_reward(r_raw, is_end):
    # Step penalty
    step_penalty = 0
    if is_end:
        # r_scale = min(r_raw / 2000.0, 1.0)
        r_scale = r_raw / 2000.0
        return r_scale + step_penalty  
    else:
        r_mid = r_raw / 2000.0  
        return r_mid + step_penalty  
    



class ELREnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 4,
    }
    
    def __init__(self, transitions, activity2idx, use_true_end_reward=False, reward_scale=1.0, render_mode=None):
        super().__init__()
        self.action2idx = {a: i for i, a in enumerate(activity2idx.keys())}
        self.idx2action = {i: a for a, i in self.action2idx.items()}
        self.transitions = transitions
        self.current_state_str = None
        self.current_state_vec = None
        self.use_true_end_reward = use_true_end_reward
        self.reward_scale = reward_scale
        
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(29,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action2idx))

        self.valid_actions_per_state = {}
        for state, actions in self.transitions.items():
            valid_action_indices = [self.action2idx[action] for action in actions.keys() 
                                  if action in self.action2idx]
            self.valid_actions_per_state[state] = valid_action_indices

    def get_valid_actions(self):
        if self.current_state_str in self.valid_actions_per_state:
            return self.valid_actions_per_state[self.current_state_str]
        else:
            return [] 

    def get_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_actions = self.get_valid_actions()
        if valid_actions:
            mask[valid_actions] = True
        return mask


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_state_str = "START"
        self.current_state_vec = encode_state("START")

        
        return self.current_state_vec.copy(), {}
    
    def step(self, action_idx):
        """
        input: action_idx(0..23)
        output: (next_obs, reward, terminated, truncated, info)
        """
        s = self.current_state_str

        valid_actions = self.get_valid_actions()
        if action_idx not in valid_actions:
            if valid_actions:
                next_vec = encode_state("END")
                reward = -1.0
                terminated = True
                truncated = False
                info = {"invalid_action": True}
                return next_vec, reward, terminated, truncated, info





        action_str = self.idx2action[action_idx]

        if s in self.transitions and action_str in self.transitions[s]:
            next_s, r_stored = self.transitions[s][action_str]
            if self.use_true_end_reward:
                reward = r_stored * self.reward_scale
            else:
                reward = scaled_reward(r_stored, next_s == "END")

            terminated = (next_s == "END")
            truncated = False
            
            next_vec = encode_state(next_s)
            self.current_state_str = next_s
            self.current_state_vec = next_vec.copy()


            
            return next_vec, reward, terminated, truncated, {}

        else:
            next_vec = encode_state("END")
            reward = -1.0
            terminated = True
            truncated = False
            return next_vec, reward, terminated, truncated, {"unknown_transition": True}
        
    def render(self):
        if self.render_mode == "human":
            print("STATE:", self.current_state_str)
            valid_actions = self.get_valid_actions()
            print(f"VALID ACTIONS: {[self.idx2action[i] for i in valid_actions]}")
    
    def close(self):
        pass



    
class ActionMaskObsWrapper(gym.Wrapper):
    """
    Wrap any discrete-action env so that observations become a Dict with:
        - "obs": the original observation
        - "mask": a 0/1 array where 1 means the action is currently legal

    Tianshou 1.1's DQNPolicy will automatically respect `obs["mask"]` for
    greedy and epsilon-greedy action selection, as well as target action
    selection in Double-DQN.
    """
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(self.env.action_space, spaces.Discrete):
            raise TypeError("ActionMaskObsWrapper requires a Discrete action space.")
        self._n_act = self.env.action_space.n

        # if the underlying environment has already returned Dict(obs, mask), then just pass through
        if isinstance(self.env.observation_space, spaces.Dict) and \
           "obs" in self.env.observation_space and "mask" in self.env.observation_space:
            self._passthrough = True
            self.observation_space = self.env.observation_space
        else:
            self._passthrough = False
            self.observation_space = spaces.Dict({
                "obs": self.env.observation_space,
                "mask": spaces.MultiBinary(self._n_act),
            })

    # ---------- helpers ----------
    def _compute_mask(self, info=None):
        """
        Priority:
        1) env.get_action_mask()(if the underlying environment provides)
        2) info["mask"](if the underlying environment puts the mask in info)
        3) info["legal_actions"](list[int])→ convert to one-hot
        4) fallback: all 1(at least one action should be legal)
        """
        import numpy as np

        if hasattr(self.env, "get_action_mask"):
            mask = self.env.get_action_mask()
        elif info is not None and "mask" in (info or {}):
            mask = info["mask"]
        elif info is not None and "legal_actions" in (info or {}):
            legal = info["legal_actions"]
            mask = np.zeros(self._n_act, dtype=np.int8)
            mask[legal] = 1
        else:
            mask = np.ones(self._n_act, dtype=np.int8)

        mask = np.asarray(mask)
        if mask.ndim == 0:
            mask = np.full(self._n_act, 1 if bool(mask) else 0, dtype=np.int8)
        if mask.dtype != np.int8 and mask.dtype != bool:
            mask = mask.astype(np.int8)
        if mask.shape != (self._n_act,):
            raise ValueError(f"mask shape {mask.shape} != ({self._n_act},)")
        return mask

    # ---------- gym API ----------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._passthrough:
            return obs, info
        mask = self._compute_mask(info)



        return {"obs": obs, "mask": mask}, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if self._passthrough:
            return obs, rew, terminated, truncated, info
        mask = self._compute_mask(info)



        return {"obs": obs, "mask": mask}, rew, terminated, truncated, info