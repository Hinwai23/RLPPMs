import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import os
import sys

def create_mdp_csv(xes_file_path, output_csv_path=None):
    """
    Creates an MDP CSV file from an XES file.
    
    Args:
        xes_file_path (str): Path to the input XES file
        output_csv_path (str): Path to the output CSV file (optional)
    
    Returns:
        pandas.DataFrame: The MDP dataframe with columns s, a, s', reward, case
    """
    
    # Read the XES file
    print(f"Reading XES file: {xes_file_path}")
    log = xes_importer.apply(xes_file_path)
    
    # Initialize list to store MDP data
    mdp_data = []
    
    # Iterate through all traces
    for trace_idx, trace in enumerate(log):
        # Get the case ID from trace attributes
        case_id = trace.attributes.get("concept:name", f"case_{trace_idx}")
        value = trace.attributes.get("amount", f"case_{trace_idx}")

        if int(value) <= 6000:
            amClass = 0
        elif int(value) > 15000:
            amClass = 2
        else:
            amClass = 1

        # state infos
        call = 0
        miss = 0
        offer = 0
        reply = 0
        fix = 0
        
        # print(f"Processing trace {trace_idx + 1}/{len(log)}: {case_id}")
        
        # Iterate through events in the trace
        for event_idx, event in enumerate(trace):
            # Current state (s): current event's concept:name
            if event.get("concept:name", "") == "START":
                current_state = event.get("concept:name", "") 

            else:
                current_state = event.get("concept:name", "") + f",{call}"  + f",{miss}" + f",{offer}" + f",{reply}" + f",{fix}"
                
            
            
            # Check if there's a next event
            if event_idx + 1 < len(trace):
                next_event = trace[event_idx + 1]
                
                # Action (a): next event's concept:name
                action = next_event.get("concept:name", "")

                if action == "W_Call_after_offer":
                    call += 1
                elif action == "W_Call_missing_information":
                    miss += 1
                elif action == "O_SENT":
                    offer += 1
                elif action == "O_SENT_BACK":
                    reply += 1
                elif action == "W_Fix_incomplete_submission":
                    fix += 1
                
                # Next state (s'): next event's concept:name 
                if next_event.get("concept:name", "") == "END":
                    next_state = "END"
                else:
                    next_state = next_event.get("concept:name", "") + f",{call}" + f",{miss}" + f",{offer}" + f",{reply}" + f",{fix}"
                
                # Reward: next event's kpi:reward
                reward = next_event.get("kpi:reward", 0)
                
                # Add the MDP transition to our data
                mdp_data.append({
                    's': current_state,
                    'a': action,
                    "s'": next_state,
                    'reward': reward,
                    'case': case_id,
                    'amount': value
                })

    
    # Create DataFrame
    df = pd.DataFrame(mdp_data)
    
    # Create output directory if it doesn't exist
    output_dir = "logs/80_20/MDP"
    os.makedirs(output_dir, exist_ok=True)
    
    # If no output path specified, create one in the MDP folder
    if output_csv_path is None:
        base_name = os.path.splitext(os.path.basename(xes_file_path))[0]
        output_csv_path = os.path.join(output_dir, f"{base_name}_mdp.csv")
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"MDP CSV saved to: {output_csv_path}")
    print(f"Total transitions: {len(df)}")
    print(f"Total cases: {df['case'].nunique()}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head(10))
    
    return df

def main():
    """
    Main function to run the MDP creator.
    Usage: python MDPCreator.py <xes_file_path>
    """
    
    if len(sys.argv) != 2:
        print("Usage: python MDPCreator.py <xes_file_path>")
        print("Example: python MDPCreatorBPI.py logs/80_20/BPI_2012_cumulative_rewards.xes")
        return
    
    xes_file_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(xes_file_path):
        print(f"Error: File {xes_file_path} does not exist!")
        return
    
    try:
        df = create_mdp_csv(xes_file_path)
        print(f"\nMDP creation completed successfully!")
        print(f"DataFrame shape: {df.shape}")
        
        # Show some statistics
        print(f"\nStatistics:")
        print(f"Unique states (s): {df['s'].nunique()}")
        print(f"Unique actions (a): {df['a'].nunique()}")
        print(f"Unique cases: {df['case'].nunique()}")
        print(f"Reward range: {df['reward'].min()} to {df['reward'].max()}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()