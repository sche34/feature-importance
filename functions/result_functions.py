from datetime import datetime
import os
import json
import numpy as np

def create_results(config, x, y, start_time):
    superpixel_width = config["superpixel_width"]

    # create actual result
    x_shape_list = []
    for dim in x.shape:
        x_shape_list.append(dim)
    results = {"timestamp": start_time.strftime("%d/%m/%Y %H:%M:%S"),
                "run_time": None,
                "feature_list": superpixel_width,
                "count per class": [np.unique(y, return_counts=True)[0].tolist(), np.unique(y, return_counts=True)[1].tolist()],
                "shapley_mode": config["shapley_mode"],
                "x-shape": x_shape_list,
                "shapley_values": [],
                "std": [],
                "script name": os.path.basename(__file__)}

    if config["shapley_mode"] == "sample_shapley":
        results["shapley_sample_percentage"] = config["mode_specific_params"]["SHAP"]["shapley_sample_percentage"]
    elif config["shapley_mode"] == "SHAP":
        results["SHAP_max_samples"] = config["mode_specific_params"]["SHAP"]["SHAP_max_samples"]
        results["SHAP_min_samples"] = config["mode_specific_params"]["SHAP"]["SHAP_min_samples"]
    return results

def update_results(results, shapley_values):
    # save the final result file
    timestamp = results['timestamp']
    start_time = datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S")
    results["run_time"] = (datetime.now() - start_time).total_seconds()

    results["shapley_values"] = []
    for feature_index in shapley_values.keys():
        values = shapley_values[feature_index]['values'].tolist()
        n_samples = shapley_values[feature_index]['n_samples']
        results["shapley_values"].append([n_samples, values])

    timestamp_json = start_time.strftime("%Y%m%d %H%M%S")
    with open(r'results\result ' + timestamp_json + '.json', 'w') as outfile:
        json.dump(results, outfile)
    return results

def save_final_results(results, shapley_values):
    # save the final result file
    timestamp = results['timestamp']
    start_time = datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S")
    results["run_time"] = (datetime.now() - start_time).total_seconds()

    results["shapley_values"] = []
    for feature_index in shapley_values.keys():
        values = shapley_values[feature_index]['values'].tolist()
        n_samples = shapley_values[feature_index]['n_samples']
        results["shapley_values"].append([n_samples, values])

    results["avg_shapley_values"] = []
    total_runs = 0
    for feature_index in shapley_values.keys():
        avg_value = np.mean(shapley_values[feature_index]['values'])
        n_samples = shapley_values[feature_index]['n_samples']
        total_runs += n_samples
        results["avg_shapley_values"].append(float(avg_value))
    results["total_runs"] = total_runs
    timestamp_json = start_time.strftime("%Y%m%d %H%M%S")
    with open(r'results\result ' + timestamp_json + '.json', 'w') as outfile:
        json.dump(results, outfile)
    print("Successfully saved results to file.")
    return results