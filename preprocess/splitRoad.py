import os
import pm4py
import time
from pm4py.objects.log.obj import EventLog, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.attributes import attributes_filter


def _months_between(start_ts, end_ts):
	"""Return rounded integer months between two datetimes."""
	delta = end_ts - start_ts
	months_float = delta.total_seconds() / (60 * 60 * 24 * 30)
	return int(round(months_float))


def addDuration(path):
	# Load and remove events with lifecycle:transition == "SCHEDULE" like splitBPI
	log = xes_importer.apply(path)
	filtered_log = attributes_filter.apply_events(
		log, ["SCHEDULE"],
		parameters={
			attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition",
			attributes_filter.Parameters.POSITIVE: False,
		}
	)

	for trace in filtered_log:
		if len(trace) == 0:
			continue
		# Insert START and END events
		first_ts = trace[0]["time:timestamp"]
		last_ts = trace[-1]["time:timestamp"]
		start_event = Event()
		start_event["concept:name"] = "START"
		start_event["task"] = "START"
		start_event["time:timestamp"] = first_ts
		start_event["kpi:reward"] = 0
		trace.insert(0, start_event)
		end_event = Event()
		end_event["concept:name"] = "END"
		end_event["task"] = "END"
		end_event["time:timestamp"] = last_ts
		end_event["kpi:reward"] = 0
		trace.append(end_event)

		# Per-event duration: months since previous original event
		previous_real_event_ts = None
		for event in trace:
			name = event.get("concept:name", "")
			if name in ["START", "END"]:
				continue
			current_ts = event.get("time:timestamp")
			if previous_real_event_ts is None:
				event["duration"] = 0
				previous_real_event_ts = current_ts
			else:
				months_gap = _months_between(previous_real_event_ts, current_ts)
				event["duration"] = int(months_gap)
				previous_real_event_ts = current_ts

		# Store total months for the trace (used by reward logic)
		total_months = _months_between(first_ts, last_ts)
		trace.attributes["duration"] = int(total_months)

	return filtered_log, path


def _safe_float(value):
	try:
		return float(value)
	except Exception:
		return None


def addRewardCumulative(log_with_duration, path):
	output_path = path.replace(".xes", "_cumulative_rewards.xes")
	for trace in log_with_duration:
		# ensure total months present
		months = trace.attributes.get("duration")
		if months is None and len(trace) > 0:
			months = _months_between(trace[0]["time:timestamp"], trace[-1]["time:timestamp"])
			trace.attributes["duration"] = int(months)

		# compute FinalAmount as max over totalPaymentAmount and amount across events
		max_amount = 0.0
		for e in trace:
			v1 = _safe_float(e.get("totalPaymentAmount"))
			v2 = _safe_float(e.get("amount"))
			if v1 is not None and v1 > max_amount:
				max_amount = v1
			if v2 is not None and v2 > max_amount:
				max_amount = v2
		trace.attributes["FinalAmount"] = float(max_amount)

		# per-event rewards: only set for Payment or Appeal events
		total_months = int(trace.attributes.get("duration", 0))
		n_events = len(trace)
		# pre-compute sums of paymentAmount
		payment_amount_sum = 0.0
		for e in trace:
			v = _safe_float(e.get("paymentAmount"))
			if v is not None:
				payment_amount_sum += v
		for idx, event in enumerate(trace):
			# ensure every event has integer kpi:reward default 0
			event["kpi:reward"] = 0
			name = event.get("concept:name", "")
			if name in ["Appeal to Judge", "Send Appeal to Prefecture"]:
				event["kpi:reward"] = int(-1)
				continue
			if name == "Payment":
				total_payment = _safe_float(event.get("totalPaymentAmount"))
				payment_amount = _safe_float(event.get("paymentAmount"))
				next_is_end = (idx + 1 < n_events and trace[idx + 1].get("concept:name", "") == "END")
				# case 1: equal and next is END
				if total_payment is not None and payment_amount is not None and abs(total_payment - payment_amount) < 1e-6 and next_is_end:
					if total_months <= 6:
						reward_val = int(3)
					elif total_months <= 12:
						reward_val = int(2)
					else:
						reward_val = int(1)
					event["kpi:reward"] = reward_val
				# case 2: not equal but next is END and sum(paymentAmount) equals totalPaymentAmount
				elif total_payment is not None and next_is_end and abs(payment_amount_sum - total_payment) < 1e-6:
					if total_months <= 6:
						reward_val = int(3)
					elif total_months <= 12:
						reward_val = int(2)
					else:
						reward_val = int(1)
					event["kpi:reward"] = reward_val
				else:
					event["concept:name"] = "Payment Partly"
					event["kpi:reward"] = int(0)

	xes_exporter.apply(log_with_duration, output_path)


def splitLog(log_path, percentage):
	log = xes_importer.apply(log_path)
	output_training = log_path.replace(".xes", "_training_{}.xes".format(percentage))
	output_testing = log_path.replace(".xes", "_testing_{}.xes".format(100 - percentage))
	traces_list = [t for t in log]
	train_len = int(len(traces_list) / 100 * percentage)
	train_log = EventLog()
	test_log = EventLog()
	for i, t in enumerate(traces_list):
		if i < train_len:
			train_log.append(t)
		else:
			test_log.append(t)
	
	xes_exporter.apply(train_log, output_training)
	xes_exporter.apply(test_log, output_testing)


if __name__ == '__main__':
	file_path = "logs/80_20/Road_Traffic_Fine_Management_Process.xes"
	print("Current working directory:", os.getcwd())
	t1 = time.time()
	log, src_path = addDuration(file_path)
	addRewardCumulative(log, src_path)
	splitLog(file_path.replace(".xes", "_cumulative_rewards.xes"), 80)
	t2 = time.time()
	print(t2 - t1)


