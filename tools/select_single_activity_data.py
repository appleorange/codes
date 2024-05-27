#select the data of a single activity from the dataset. 
#the inputs are from a dir of json files, each file contains a number of attributes, including activities, posture, ID. 
#the program selects the data with only one activity, and output the data to a new dir of json files and image files
#the output jason file transform the activities into an integer by the category_info dictionary. For example, 'Drink.Frombottle' is transformed into 0.

import os
import json, string, sys
from glob import glob
import numpy as np
import re
import matplotlib.pyplot as plt

category_info = {'Drink.Frombottle': 0, 'Drink.Fromcup': 1, 'Eat.Useutensil': 2, 'Eat.Snack': 3, 'Use.Tablet': 4,
                 'Use.Phone': 5, 'Call.Onphone': 6, 'Use.Computer': 7, 'Type.Onkeyboard': 8, 'Use.Switch': 9,
                 'Read': 10, 'Write': 11, 'Play.Cards': 12, 'Play.Chess': 13, 'Play.Lego': 14,
                 'Play.Boardgame': 15, 'Cook.Cut': 16, 'Cook.Usestove': 17, 'Cook.Useoven': 18, 'Cook.Usemicrowave': 19,
                 'Use.Coffeemachine': 20, 'Use.Kettle': 21, 'Use.Refrig': 22, 'Wash.Hands': 23, 'Wash.Dishes': 24,
                 'Use.Sink': 25, 'Use.Shelf': 26, 'Use.Drawer': 27, 'Use.Dishwasher': 28, 'Use.Mop': 29,
                 'Use.Vaccum': 30, 'Nap': 31, 'Use.Gamecontroller': 32, 'Watch.TV': 33, 'Exercise': 34,
                 'Lay.Onbed': 35, 'Getup': 36, 'Draw.Curtain': 37, 'Move.Object': 38, 'Use.Tap': 39,
                 'Use.Switches': 40, 'Cook.Stir': 41, 'Use.Mouse': 42, 'Enter': 43, 'Leave': 44,
                 'Stand': 45, 'Sit': 46, 'lie flat': 47, 'lie on the side': 48, 'go prone': 49,
                 'squat': 50, '101': 51, '102': 52, '103': 53, '104': 54,
                 '105': 55, '106': 56, '107': 57, '108': 58, '109': 59,
                 '110': 60, '111': 61, '112': 62, '113': 63, '114': 64,
                 '115': 65, '116': 66, '117': 67, '118': 68, '119': 69,
                 '199': 70, 'male': 71, 'female': 72}

def insert_string_list_to_dict(string_list, dictionary):
    # Convert the list of strings into a single string
    combined_string = ' '.join(string_list)

    # Check if the combined string already exists in the dictionary
    if combined_string not in dictionary:
        # If it doesn't exist, insert it into the dictionary with a default value
        dictionary[combined_string] = 0
    else:
        # If it does exist, increment the value
        dictionary[combined_string] += 1

    return dictionary

def plotData(label_histgram, title):
    sorted_dict = dict(sorted(label_histgram.items(), key=lambda item: item[1], reverse=True))

    # Extract keys and values from the dictionary
    activities = list(sorted_dict.keys())
    occurences = list(sorted_dict.values())

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(activities, occurences)

    # Add labels and title
    ax.set_xlabel('activities')
    ax.set_ylabel('occurences')
    ax.set_title(title)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

num_files = 0
num_single_activies = 0

os.system(f"rm -rf data/*")  
os.system(f"mkdir -p data/train/image")
os.system(f"mkdir -p data/train/label")
os.system(f"mkdir -p data/test/image")
os.system(f"mkdir -p data/test/label")  

for name in glob(os.path.join("cropped_label", "*.json")):
    num_files += 1
    basename = name.split('/')[-1]
    if num_single_activies < 30000:
        if num_files % 100 == 0:
            print(name + " " + basename)
    else:
        break
    with open(name) as f:
        data = json.load(f)
        num_activities = 0
        valid_activity = ""

        # Check there is only one activty and the activity is not None
        for activity in data['activities']:
            activity = activity.strip()
            #print(f"!!!!activity = {activity} valid_activity = {valid_activity}")
            if len(activity) and ("None" != activity):
                num_activities += 1
                valid_activity = activity

        #print(f"num_activities = {num_activities} valid_activity = {valid_activity}")
        if num_activities == 1:
            num_single_activies += 1

            dir_name = ""
            if (num_single_activies % 100 < 80):
                dir_name = "data/train"
            else:
                dir_name = "data/test"

            #write a new json file with the activity transformed into an integer
            new_name = os.path.join(dir_name, "label", basename)
            with open(new_name, 'w') as f:
                data['activity_label'] = category_info[valid_activity]
                data['activity_name'] = valid_activity
                json.dump(data, f)
            #copy the image file to the new single_activity_image dir
            image_name = basename[:-4] + "jpg"
            os.system(f"cp cropped_image/{image_name} {dir_name}/image/{image_name}")  
        else:
            if (num_files % 100 == 0):
                print(f"!skip: multiple activities {num_activities} vs. data['num_activities'] {data['num_activities']}")

print(f"total files {num_files}, single activity files {num_single_activies}")