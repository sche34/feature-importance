import os
import json

for class_num in range(5,10):
    with open("config.json", "r") as f:
        config = json.load(f)
    config["classes"] = [class_num]
    with open("config.json", "w") as f:
        json.dump(config, f)
    os.system('python main.py')