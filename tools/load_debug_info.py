import torch
from glob import glob
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_training_results

type_of_files = ['vaccuracy', 'vloss', 'accuracy', 'loss']
vaccuracy_merged_list = []  # Initialize an empty dictionary to merge all dictionaries into
vloss_merged_list = []  # Initialize an empty dictionary to merge all dictionaries into
accuracy_merged_list = []  # Initialize an empty dictionary to merge all dictionaries into
loss_merged_list = []  # Initialize an empty dictionary to merge all dictionaries into

for file_type in type_of_files:
    file_pattern = f'models/{file_type}_history_model_resnet18_20240720_*.pt'
    merged_list = []  # Initialize an empty dictionary to merge all dictionaries into

    for name in glob(file_pattern):
        print(name)
        loaded_list = torch.load(name)
        merged_list += loaded_list
        print(loaded_list)

    print(f"Merged {file_type} list:", merged_list)
    # assign the merged_list to the corresponding variable based on file_type
    if file_type == 'vaccuracy':
        vaccuracy_merged_list = merged_list
    elif file_type == 'vloss':
        vloss_merged_list = merged_list
    elif file_type == 'accuracy':
        accuracy_merged_list = merged_list
    elif file_type == 'loss':
        loss_merged_list = merged_list

print("Merged vaccuracy list:", vaccuracy_merged_list)
print("Merged vloss list:", vloss_merged_list)
print("Merged accuracy list:", accuracy_merged_list)
print("Merged loss list:", loss_merged_list)

# Call the plot_training_results function with the merged lists
plot_training_results(accuracy_merged_list, vaccuracy_merged_list, loss_merged_list, vloss_merged_list, "Training Loss and Accuracy History", "loss_history.png")

