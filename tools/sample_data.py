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


#num_bins = 46
test_label_histgram = {}
eval_label_histgram = {}
train_label_histgram = {}
num_files = 0 

train_pattern = r"^0000[0]\d+.json$"
test_pattern = r"^0000[2][4-6]\d+.json$"
eval_pattern = r"^0000[2][0-3]\d+.json$"


for name in glob(os.path.join("cropped_label", "*.json")):
    num_files += 1
    basename = name.split('/')[-1]
    if num_files <100:
        print(name + " " + basename)
    with open(name) as f:
        data = json.load(f)
        for activity in data['activities']:
             if len(activity.strip()) and ("None" not in activity):
                activity = activity.strip()
                if re.match(test_pattern, basename):
                    test_label_histgram[activity] = test_label_histgram.get(activity, 0) + 1 
                elif re.match(eval_pattern, basename):
                    eval_label_histgram[activity] = eval_label_histgram.get(activity, 0) + 1
                elif re.match(train_pattern, basename):
                    train_label_histgram[activity] = train_label_histgram.get(activity, 0) + 1
                else:
                    continue

print("preparing to plot label histogram")
plotData(train_label_histgram, "training_labels")
plotData(eval_label_histgram, "eval_labels")
plotData(test_label_histgram, "test_labels")