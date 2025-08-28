## RLPPMs: Reinforcement Learning for Process Performance Management

This repository contains data preprocessing, environment construction, training, tuning, and evaluation code for learning reinforcement learning policies on business process logs. It supports multiple datasets and two task families:

- EL (Event Log) tasks
- RTFM (Road Traffic Fine Management) tasks

Environments are built from XES logs via MDP extraction. Agents (DQN, PPO, Rainbow) are trained and evaluated with tianshou.


## Environment

- OS: macOS (tested)
- Python: 3.10+ recommended
- Key libraries: pm4py, pandas, numpy, torch, tianshou, gymnasium

Install core dependencies with pip:

```bash
pip install pm4py pandas numpy torch tianshou gymnasium
```


## Data Preprocessing

Location: `preprocess/`

### BPI (reference)
- `splitBPI.py`
  - Adds START/END events
  - Computes per-activity durations
  - Adds cumulative rewards
  - Splits 80/20 into `<name>_training_80.xes` and `<name>_testing_20.xes`

### RTFM (Road Traffic Fine Management)
- `splitRoad.py`
  - Filters out events where `lifecycle:transition == "SCHEDULE"`
  - Inserts START and END for each trace; sets `kpi:reward = 0` on both
  - Per-event duration: months since previous original event (first original event duration = 0), rounded to integer
  - Adds trace-level `FinalAmount` = max over all events' `totalPaymentAmount` and `amount`
  - Rewards:
    - `Appeal to Judge`, `Send Appeal to Prefecture`: reward = -1
    - `Payment`: if it is the final payment (the event is immediately followed by `END`) and full paid condition holds, reward is based on total months (≤6: 3; ≤12: 2; >12: 1), set on the `Payment` event itself
    - If `totalPaymentAmount == paymentAmount` but not followed by END, rename to `Payment Partly` and reward = 0
    - If not equal but followed by END and sum of all `paymentAmount` equals `totalPaymentAmount`, treat as full paid; reward as above
  - Ensures every event has integer `kpi:reward` (default 0)
  - Outputs `<name>_cumulative_rewards.xes` and the 80/20 split

Run RTFM preprocessing:

```bash
python3 preprocess/splitRoad.py
```

### MDP Creation
- `preprocess/MDPCreatorBPI.py`: Creates MDP CSV for EL/BPI setting. Columns: `s, a, s', reward, case, amount` (EL).
- `preprocess/MDPCreatorRTFM.py`: Creates MDP CSV for RTFM. Columns: `s, a, s', reward, case`.
  - State encoding in CSV (string): `"<concept:name>,<months2>,<amClass>"`
  - `months2 = floor(duration / 2)`, `amClass = 1 if FinalAmount >= 50 else 0`

Run RTFM MDP creation (defaults to training/testing XES):

```bash
python3 preprocess/MDPCreatorRTFM.py
```

MDP CSVs are saved under `preprocess/logs/80_20/MDP/`.


## Gym Environments

### EL_env/csv_to_gym_EL.py
- Builds an EL environment with:
  - Observation: 29-dim vector (24 action one-hot + 5 counters)
  - Transitions:
    - Train: most-common next state per (s,a), reward averaged over samples
    - Test: real next state and reward
    - Merge: train preferred, test fills gaps

### RTFM_env/csv_to_gym_RTFM.py
- Builds the RTFM environment with empirical diversity:
  - Observation: `(len(actions) + 2)` dims: action one-hot, plus `[months2, amClass]`
  - Transitions: keep all observed `(s', r)` pairs for each `(s,a)`; step() samples uniformly from observed pairs (frequency-weighted)
  - Utilities exposed: `all_transitions`, `activity2idx`, `encode_state`

Example use:

```python
from RTFM_env.csv_to_gym_RTFM import RTFMEnv, activity2idx, all_transitions
env = RTFMEnv(all_transitions, activity2idx)
obs, info = env.reset()
```


## Training

Benchmarks are under:
- `benckmarkRTFM/` for RTFM (DQN.py, PPO.py, Rainbow.py)
- `benchmarkEL/` and `benchmarkBPI/` for EL/BPI

Ensure the model architecture (state dim, action dim) matches the environment you use.


## Hyperparameter Tuning

Scripts: `tuning/` with per-algorithm tuning entries and plots under `tuning/plot/`.
TensorBoard logs are under `logs/` and `training_logs/`.


## Evaluation

Two evaluation styles exist in each task family:

### evaluation_*/*_performance_1.py
- Given learned policy and test MDP CSV, compute per-case metrics:
  - Optimal policy agreement ratio per case
  - KPI as sum of rewards
  - For RTFM: `full Payment` ratio = fraction of traces containing at least one `Payment` action (per set: ALL / OPTIMAL / NON-OPTIMAL)

RTFM examples:

```bash
python3 evaluation_RTFM/evaluate_dqn_performance_1.py
python3 evaluation_RTFM/evaluate_rainbow_performance_1.py
python3 evaluation_RTFM/evaluate_ppo_performance_1.py
```

### evaluation_*/*_performance_2.py (Prefix Simulation)
- Baseline `a` = sum of all rewards in the test CSV
- For each case and prefix p (default RTFM scripts: 1–5), start from the p-th state and roll out actions using the learned policy; next state is chosen as the most representative (mode) among observed `(s', r)` for `(s,a)`; reward uses that mode's average reward. Sum over cases gives `b`.
- Report `avg_delta = (b - a) / #cases` for each prefix.

RTFM examples:

```bash
python3 evaluation_RTFM/evaluate_dqn_performance_2.py
python3 evaluation_RTFM/evaluate_rainbow_performance_2.py
python3 evaluation_RTFM/evaluate_ppo_performance_2.py
```


## Reproducibility & Tips

- Paths: scripts default to data under `preprocess/logs/80_20/` and outputs under `preprocess/logs/80_20/MDP/`
- Randomness: RTFM environment samples transitions from empirical pairs; fix NumPy/PyTorch seeds if you need deterministic rollouts
- State shapes:
  - EL: `len(actions) + 5` (implemented as 29 with 24 actions)
  - RTFM: `len(actions) + 2`
- Model checkpoints: store as `*_rtfm_best.pth` or `*_el_best.pth` consistent with evaluators


## Project Structure (key parts)

```
preprocess/
  splitBPI.py, splitRoad.py, MDPCreatorBPI.py, MDPCreatorRTFM.py
  logs/80_20/*.xes and MDP/*.csv
EL_env/ csv_to_gym_EL.py
RTFM_env/ csv_to_gym_RTFM.py
benchmarkEL/, benchmarkBPI/, benckmarkRTFM/
evaluation_RTFM/ evaluate_*_performance_{1,2}.py
evaluations_EL/, evaluations_BPI/
tuning/
```


## Citation

If this codebase supports your research, please cite accordingly (add your thesis/paper information here).


