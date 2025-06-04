import pandas as pd


def extract_ints_from_state(state_str):
    parts = state_str.split(',')
    return [int(x) for x in parts[1:]]  # 跳过第一个label

def find_max_ints(states):
    max_vals = [float('-inf')] * 5
    for s in states:
        if s in ("START", "END"):
            continue  
        ints = extract_ints_from_state(s)
        for i in range(5):
            if ints[i] > max_vals[i]:
                max_vals[i] = ints[i]
    return max_vals

def check_states_actions(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_states = set(train_df['s']).union(set(train_df["s'"]))
    train_actions = set(train_df['a'])

    test_states = set(test_df['s']).union(set(test_df["s'"]))
    test_actions = set(test_df['a'])

    train_max = find_max_ints(train_states)
    test_max = find_max_ints(test_states)
    print(f"Train states max ints: {train_max}")
    print(f"Test states max ints: {test_max}")

    missing_states = test_states - train_states
    missing_actions = test_actions - train_actions

    if not missing_states and not missing_actions:
        print("All states and actions in the test set are present in the training set.")
    else:
        if missing_states:
            print(f"{len(train_states) / (len(train_states) + len(missing_states)) * 100}% states are covered in the training set.")
            # for s in missing_states:
            #     print(f"  - {s}")
        if missing_actions:
            print(f"{len(train_actions) / (len(train_actions) + len(missing_actions)) * 100}% actions are covered in the training set.")
            # for a in missing_actions:
            #     print(f"  - {a}")




if __name__ == "__main__":
    check_states_actions("logs/80_20/MDP/BPI_2012_cumulative_rewards_training_80_mdp.csv", "logs/80_20/MDP/BPI_2012_cumulative_rewards_testing_20_mdp.csv")