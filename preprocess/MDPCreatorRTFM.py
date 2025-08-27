import os
import sys
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer


def create_mdp_csv(xes_file_path, output_csv_path=None):
	"""
	Create MDP CSV from Road Traffic Fine Management Process XES.

	MDP columns: s, a, s', reward, case
	State (except START/END): "<concept:name>,<months2>,<amClass>"
	- months2 = floor(event.duration / 2)
	- amClass = 1 if trace.FinalAmount >= 50 else 0
	"""

	print(f"Reading XES file: {xes_file_path}")
	log = xes_importer.apply(xes_file_path)

	mdp_rows = []

	for trace_idx, trace in enumerate(log):
		case_id = trace.attributes.get("concept:name", f"case_{trace_idx}")
		final_amount = trace.attributes.get("FinalAmount", 0.0)
		try:
			final_amount = float(final_amount)
		except Exception:
			final_amount = 0.0
		amClass = 1 if final_amount >= 50.0 else 0

		for i, event in enumerate(trace):
			name = event.get("concept:name", "")
			# current state s
			if name == "START":
				s_state = "START"
			else:
				duration = int(event.get("duration", 0))
				months2 = duration // 2
				s_state = f"{name},{months2},{amClass}"

			# transition only if next event exists
			if i + 1 < len(trace):
				next_event = trace[i + 1]
				action = next_event.get("concept:name", "")
				# next state s'
				if action == "END":
					s_prime = "END"
				else:
					next_duration = int(next_event.get("duration", 0))
					next_months2 = next_duration // 2
					s_prime = f"{action},{next_months2},{amClass}"
				# reward from next event
				reward = int(next_event.get("kpi:reward", 0))

				mdp_rows.append({
					's': s_state,
					'a': action,
					"s'": s_prime,
					'reward': reward,
					'case': case_id
				})

	# DataFrame
	df = pd.DataFrame(mdp_rows, columns=['s', 'a', "s'", 'reward', 'case'])

	# Output path handling
	base_dir = os.path.dirname(os.path.abspath(__file__))
	output_dir = os.path.join(base_dir, "logs/80_20/MDP")
	os.makedirs(output_dir, exist_ok=True)
	if output_csv_path is None:
		base_name = os.path.splitext(os.path.basename(xes_file_path))[0]
		output_csv_path = os.path.join(output_dir, f"{base_name}_mdp.csv")

	df.to_csv(output_csv_path, index=False)
	print(f"MDP CSV saved to: {output_csv_path}")
	print(f"Total transitions: {len(df)} | Total cases: {df['case'].nunique()}")
	return df


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	default_files = [
		os.path.join(base_dir, "logs/80_20/Road_Traffic_Fine_Management_Process_cumulative_rewards_training_80.xes"),
		os.path.join(base_dir, "logs/80_20/Road_Traffic_Fine_Management_Process_cumulative_rewards_testing_20.xes"),
	]

	args = sys.argv[1:]
	files = args if len(args) > 0 else default_files
	for path in files:
		if not os.path.exists(path):
			print(f"Warn: {path} not found, skipping")
			continue
		df = create_mdp_csv(path)
		# Statistics (similar to BPI)
		print("\nStatistics:")
		print(f"DataFrame shape: {df.shape}")
		print(f"Unique states (s): {df['s'].nunique()}")
		print(f"Unique actions (a): {df['a'].nunique()}")
		print(f"Unique cases: {df['case'].nunique()}")
		if not df.empty:
			print(f"Reward range: {df['reward'].min()} to {df['reward'].max()}")
			print("\nSample data:")
			print(df.head(10))


if __name__ == "__main__":
	main()


