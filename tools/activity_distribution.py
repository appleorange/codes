#iterate the json file from a given directory. The activity is a label from 0 to 45. The function should return a dictionary with 
#the distribution of the activities in the json files. 
#for example, there are 100 json files. 10 of them have activity 0, 20 of them have activity 1, 70 of them have activity 2.
#the distribution of activities is {0: 10%, 1: 20%, 2: 70%}.

import os
import json
import re
from glob import glob
import numpy as np

import matplotlib.pyplot as plt

def activity_distribution(directory):
    activity_dict = {}
    num_files = 0
    for name in glob(os.path.join(directory, "*.json")):
        num_files += 1
        with open(name) as f:
            data = json.load(f)
            activity = data['activity_label']
            activity_dict[activity] = activity_dict.get(activity, 0) + 1
    activity_dict = {k: v/num_files for k, v in activity_dict.items()}
    return activity_dict

# sort the dictionary based on the key in the ascending order. print the dictionary value as %.
def print_activity_distribution(directory):
    activity_dict = activity_distribution(directory)
    sorted_dict = dict(sorted(activity_dict.items(), key=lambda item: item[0]))
    for key, value in sorted_dict.items():
        print(f"{key}: {value*100:.2f}%")
    return sorted_dict

print_activity_distribution("data/test/label")

