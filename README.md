# Reinforcement Learning for Prescriptive Process Monitoring

This project explores the application of Reinforcement Learning (RL) to recommend the "next best activity" in business processes, aiming to optimize a defined Key Performance Indicator (KPI). It leverages a previously established architecture to train and evaluate RL agents on four different event logs from synthetic and real-world domains.

## Project Overview

The core objective is to investigate how RL-based policies can be used for prescriptive process monitoring. The project workflow is as follows:
1.  **Environment Creation**: Raw event logs are preprocessed and transformed into Markov Decision Processes (MDPs), which are then wrapped in custom `gymnasium`-compatible environments.
2.  **Agent Training**: Three popular RL algorithms—DQN, Rainbow, and PPO—are trained on these environments using the `tianshou` framework.
3.  **Policy Evaluation**: The performance of the learned policies is assessed using two distinct evaluation methodologies to determine their effectiveness in improving KPIs.

## Datasets

The study utilizes four event logs:
1.  **Synthetic EL (`_EL_`)**: A synthetic log representing a standard business process.
2.  **Synthetic ELR (`_ELR_`)**: A variant of the synthetic log with rarer optimal traces.
3.  **BPI 2012 (`_BPI_`)**: A real-world event log from a Dutch financial institute concerning a loan application process.
4.  **Road Traffic Fine Management (`_RTFM_`)**: A real-world event log detailing the process of managing road traffic fines in Italy.

## Methodology

### RL Algorithms
-   **Deep Q-Network (DQN)**
-   **Rainbow DQN**
-   **Proximal Policy Optimization (PPO)**

### Evaluation Methods
Two evaluation methods are employed to analyze the performance of the trained agents from different perspectives:

1.  **Retrospective Policy Conformance Analysis**: This method performs a retrospective analysis on completed traces from the test log. It classifies each trace as conforming or non-conforming to the agent's optimal policy. By comparing the average KPI of the conforming group against the overall average, we can quantify the effectiveness of the learned policy. A higher KPI for the conforming group indicates that the policy has successfully identified superior operational strategies.

2.  **Prescriptive Recommendation Simulation**: This method evaluates the policy's utility as a real-time recommendation tool for ongoing cases. It simulates the completion of partial traces by following the agent's recommendations from a specific point (prefix) onwards. The analysis measures the "delta KPI"—the potential improvement in the final outcome—for various prefix lengths, revealing when the agent's guidance is most impactful.

## Project Structure

```
.
├── benchmark*               # Scripts for training final models
│   ├── DQN.py
│   ├── PPO.py
│   └── Rainbow.py
├── evaluations_*            # Scripts for evaluating trained models
│   ├── evaluate_*_1.py      # Evaluation Method 1
│   └── evaluate_*_2.py      # Evaluation Method 2
├── *_env                    # Custom gymnasium environment wrappers
│   └── csv_to_gym_*.py
├── preprocess/              # Scripts for data preprocessing and MDP creation
│   ├── MDPCreator*.py
│   └── split*.py
├── tuning/                  # Scripts for hyperparameter tuning
│   ├── tune_*.py
│   └── plot/
├── logs/                    # TensorBoard logs for tuning and training
├── *.pth                    # Saved model weights
└── README.md
```

## How to Run

### Prerequisites
- Python 3.8+
- Create a virtual environment and install the required packages:
  ```bash
  pip install torch tianshou pandas numpy gymnasium pm4py
  ```

### Execution Pipeline
The project is run in a sequence of steps for each dataset.

**Step 1: Preprocess Data**
Convert the raw event logs into an MDP format.
```bash
# Example for BPI 2012 dataset
python preprocess/splitBPI.py
python preprocess/MDPCreatorBPI.py
```

**Step 2: Hyperparameter Tuning (Optional)**
Run tuning scripts to find the best hyperparameters for each algorithm.
```bash
# Example for PPO on the BPI 2012 dataset
python tuning/tune_PPO.py
```

**Step 3: Train the Final Models**
Train the agents using the determined hyperparameters and save the final models.
```bash
# Example for PPO on the BPI 2012 dataset
python benchmarkBPI/PPO.py
```

**Step 4: Evaluate the Models**
Run the evaluation scripts to assess the performance of the trained policies.
```bash
# Example for PPO on the BPI 2012 dataset
python evaluations_BPI/evaluate_ppo_performance_1.py
python evaluations_BPI/evaluate_ppo_performance_2.py
```

## Key Findings

- **Policy Effectiveness**: The evaluation results confirm that RL policies can successfully learn and recommend actions that lead to significant KPI improvements.
- **Value of Early Intervention**: The prescriptive analysis consistently shows that recommendations are most impactful when provided during the early stages of a process.
- **Real-World vs. Synthetic Data**: A notable performance gap exists between models trained on synthetic versus real-world logs. The complexity and noise in real-world data present significant challenges, sometimes leading to policies that can be ineffective or even detrimental if not carefully validated.

## Future Work

The ultimate goal is to translate this research into a production-ready, real-time recommendation agent. This requires deep collaboration with business experts to:
-   Gather extensive domain knowledge to engineer a high-fidelity training environment.
-   Develop a more sophisticated state representation that captures a wider range of contextual information.
-   Design a nuanced reward function that accurately reflects complex business objectives and trade-offs.
