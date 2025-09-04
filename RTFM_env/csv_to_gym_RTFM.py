import pandas as pd
import numpy as np
from gymnasium import spaces
import gymnasium as gym
import copy
from collections import Counter
from gymnasium import Wrapper


# Load RTFM MDP CSVs
train_df = pd.read_csv('preprocess/logs/80_20/MDP/Road_Traffic_Fine_Management_Process_cumulative_rewards_training_80_mdp.csv')
test_df = pd.read_csv('preprocess/logs/80_20/MDP/Road_Traffic_Fine_Management_Process_cumulative_rewards_testing_20_mdp.csv')

# Actions dictionary from training (unique action symbols)
all_activities = sorted(train_df['a'].unique().tolist())
activity2idx = {act: i for i, act in enumerate(all_activities)}


# Build raw lists
raw_rewards = {}      # raw_rewards[s][a] = [r1, r2, r3, ...]
raw_next_states = {}  # raw_next_states[s][a] = [s'_1, s'_2, ...]

for _, row in train_df.iterrows():
	s  = row['s']
	a  = row['a']
	sp = row["s'"]
	r  = float(row['reward'])
	raw_rewards.setdefault(s, {}).setdefault(a, []).append(r)
	raw_next_states.setdefault(s, {}).setdefault(a, []).append(sp)


# Preserve empirical diversity: keep all observed (s', r) samples per (s,a)
train_samples = {}  # train_samples[s][a] = [(sp1, r1), (sp2, r2), ...]
for s, adict in raw_next_states.items():
	for a, sp_list in adict.items():
		r_list = raw_rewards[s][a]
		pairs = [(sp_list[i], r_list[i]) for i in range(len(sp_list))]
		train_samples.setdefault(s, {})[a] = pairs


train_transitions = train_samples

# Test samples: collect all observed (s', r)
test_samples = {}
for _, row in test_df.iterrows():
	s  = row['s']
	a  = row['a']
	sp = row["s'"]
	r  = float(row['reward'])
	test_samples.setdefault(s, {}).setdefault(a, []).append((sp, r))


# Merge: prefer train samples; fill gaps with test samples
all_samples = copy.deepcopy(train_samples)
for s, adict in test_samples.items():
	for a, pair_list in adict.items():
		if s not in all_samples:
			all_samples[s] = {}
		if a not in all_samples[s]:
			all_samples[s][a] = pair_list

# Expose transitions variable name for compatibility
all_transitions = all_samples


def encode_state(state_str):
	"""
	Encode state to a 15-dim vector: 13 one-hot actions + months2 + amClass.
	START/END -> zero vector.
	"""
	vec = np.zeros((len(activity2idx) + 2,), dtype=np.float32)
	if state_str == "START" or state_str == "END":
		return vec
	parts = state_str.split(',')
	if len(parts) < 3:
		return vec
	activity = parts[0]
	try:
		months2 = float(parts[1])
	except Exception:
		months2 = 0.0
	try:
		am_class = float(parts[2])
	except Exception:
		am_class = 0.0
	if activity in activity2idx:
		idx = activity2idx[activity]
		vec[idx] = 1.0
	# append months2 and amClass at the end
	vec[-2] = months2 / 70.0
	vec[-1] = am_class
	return vec


class RTFMEnv(gym.Env):
	metadata = {
		"render_modes": ["human"],
		"render_fps": 4,
	}

	def __init__(self, transitions, activity2idx, render_mode=None):
		super().__init__()
		self.action2idx = {a: i for i, a in enumerate(activity2idx.keys())}
		self.idx2action = {i: a for a, i in self.action2idx.items()}
		self.transitions = transitions
		self.current_state_str = None
		self.current_state_vec = None
		self.render_mode = render_mode

		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(activity2idx) + 2,), dtype=np.float32)
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
		input: action_idx
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
			pairs = self.transitions[s][action_str]
			choice_idx = np.random.randint(len(pairs))
			next_s, r_stored = pairs[choice_idx]
			reward = float(r_stored)
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


