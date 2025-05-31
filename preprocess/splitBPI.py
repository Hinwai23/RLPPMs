import os
import numpy as np
import pm4py
import random
import datetime
import time
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log.importer.xes import importer as xes_importer


AMOUNT_LABEL = "amount"

EVENT_NAME_CORRECTIONS = {
    "W_Fix_incoplete_submission": "W_Fix_incomplete_submission",
}

def splitLog(log_path, percentage):
	# tracefilter_log_pos = pm4py.read_xes(log_path)
	tracefilter_log_pos = xes_importer.apply(log_path)
	output_training = log_path.replace(".xes", "_training_{}.xes".format(percentage))
	output_testing = log_path.replace(".xes", "_testing_{}.xes".format(100-percentage))

	traces_list = [x for x in tracefilter_log_pos]
	train_l = int(len(traces_list) / 100 * percentage)
	train_log = EventLog()
	test_log = EventLog()
	train_acc_cases_count = 0
	train_decl_cases_count = 0
	train_others_cases_count = 0
	test_acc_cases_count = 0
	test_decl_cases_count = 0
	test_others_cases_count = 0

	for i, t in enumerate(traces_list):
		print(type(t))
		events = [e["concept:name"] for e in t]
		if i < train_l:
			train_log.append(t)
			if "O_ACCEPTED" in events:
				train_acc_cases_count += 1
			elif "O_DECLINED" in events:
				train_decl_cases_count += 1
			else:
				train_others_cases_count += 1
		else:
			test_log.append(t)
			if "O_ACCEPTED" in events:
				test_acc_cases_count += 1
			elif "O_DECLINED" in events:
				test_decl_cases_count += 1
			else:
				test_others_cases_count += 1

	xes_exporter.apply(train_log, output_training)
	xes_exporter.apply(test_log, output_testing)

	print("Train log:\nTotal cases count: {}\nAccepted cases count: {}\n Declined cases count: {}\n Other cases count:{}".format(len(train_log),
																																 train_acc_cases_count,
																																 train_decl_cases_count,
																											 train_others_cases_count))
	print("Test log:\nTotal cases count: {}\nAccepted cases count: {}\n Declined cases count: {}\n Other cases count:{}".format(len(test_log),
																																 test_acc_cases_count,
																																 test_decl_cases_count,
																																 test_others_cases_count))



def addDuration(path):
	log = pm4py.read_xes(path)
	begin_event = Event()
	begin_event["concept:name"] = ""

	tracefilter_log = attributes_filter.apply_events(log, ["SCHEDULE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: False})

	for trace in tracefilter_log:
		start = Event()
		start["concept:name"] = "START"
		start["task"] = "START"
		start["time:timestamp"] = trace[0]["time:timestamp"]
		trace.insert(0, start)
		end = Event()
		end["concept:name"] = "END"
		end["task"] = "END"
		end["time:timestamp"] = trace[-1]["time:timestamp"]
		trace.append(end)
		last_event = Event()
		last_event["concept:name"] = ""
		for event in trace:

			current_name = event["concept:name"]
			if current_name in EVENT_NAME_CORRECTIONS:
				corrected_name = EVENT_NAME_CORRECTIONS[current_name]
				event["concept:name"] = corrected_name
				
			if event["concept:name"] not in ["START", "END"]:
				if event["concept:name"].startswith("W_") and event["lifecycle:transition"] in ["START", "start"]:
					begin_event = event
				if event["concept:name"] == begin_event["concept:name"] and event["lifecycle:transition"] in ["COMPLETE", "complete"]:
					duration = event["time:timestamp"] - begin_event["time:timestamp"]
					event["concept:name"] = "TO_REMOVE"
					if event["time:timestamp"].day is not begin_event["time:timestamp"].day:
						begin_event["duration"] = 2000
					else:
						begin_event["duration"] = duration.total_seconds()
					begin_event = Event()
					begin_event["concept:name"] = ""
				if last_event["concept:name"] == event["concept:name"] and event["concept:name"] != "TO_REMOVE":
					if "duration" in last_event.keys() and "duration" in event.keys():
						event["duration"] = event["duration"] + last_event["duration"]
					idx = [x for x, e in enumerate(trace) if e == last_event][0]
					trace[idx]["concept:name"] = "TO_REMOVE"


			last_event = event

	tracefilter_log_pos_2 = attributes_filter.apply_events(tracefilter_log, ["TO_REMOVE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	return tracefilter_log_pos_2, path



def addRewardCumulative(tracefilter_log_pos_2, path):
	output_path = path.replace(".xes", "_cumulative_rewards.xes")
	for trace in tracefilter_log_pos_2:
		objective_event = False
		cumulative_reward = 0
		if AMOUNT_LABEL in trace.attributes.keys():
			amount = int(trace.attributes[AMOUNT_LABEL])
		else:
			amount = 0
		for event in trace:
			event["amount"] = amount
			if 'duration' not in event.keys():
				event['duration'] = 0
			reward = 0
			if 'duration' in event.keys() and event["concept:name"] not in ["START", "END"]:
				reward -= 0.01 * event['duration']
			if event["concept:name"] == "O_ACCEPTED":
				objective_event = True
			if event["concept:name"] == "END" and objective_event:
				reward += 0.15 * amount
			cumulative_reward += reward
			event["kpi:reward"] = round(reward, 2)

	xes_exporter.apply(tracefilter_log_pos_2, output_path)



if __name__ == '__main__':
	file_names = ['BPI_2012']
	for file_name in file_names:
		print("Current working directory:", os.getcwd())
		print('file:', file_name)
		t1 = time.time()
		log, path = addDuration("env/logs/80_20/" + file_name + '.xes')
		addRewardCumulative(log, path)
		splitLog("env/logs/80_20/" + file_name + "_cumulative_rewards.xes", 80)
		t2 = time.time()
		print(t2-t1)