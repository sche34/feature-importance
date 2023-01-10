import json
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from datetime import datetime

from functions import data_sampling, result_functions
from explainer import SHAP

# load the model, data, and feature list. Here you can change the class numbers and whether to send an email
model = keras.models.load_model("modelLevel2.h5")
json_object = open('config.json')
config = json.load(json_object)

# get the data
data_mode = config["data_mode"]
if data_mode == "full":
	x, y, file_paths = data_sampling.get_full_classes(config["classes"])
elif data_mode == "partial":
	n_data_samples = config["mode_specific_params"][data_mode]["n_data_samples"]
	x, y, file_paths = data_sampling.sample_classes(config["classes"], n_data_samples)
else:
	raise ValueError("sample_method in config must be either 'full' or 'partial'")

# Some stuff that is needed before starting to calculate the shapley values
start_time = datetime.now()
timestamp_json = start_time.strftime("%Y%m%d %H%M%S")
results = result_functions.create_results(config, x, y, start_time)
os.environ["warning_displayed"] = "False"

with tf.device('/CPU:0'):
	if config["shapley_mode"] == "SHAP":
		SHAP(x, y, model, config, results)
	else:
		raise ValueError("shapley mode not recognized, should be 'SHAP'")

# send an email to the user that script is done
# if config["send_mail"] == True:
# 	utils.send_email('timo.scheidel@live.nl', results, printing=True)