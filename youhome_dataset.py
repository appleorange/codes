import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
import pickle
from pdb import set_trace as stop
from PIL import Image
import json, string, sys
import torchvision.transforms.functional as TF
import random
import csv
# from data_utils import get_unk_mask_indices
from glob import glob


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


class YouHomeDataset(torch.utils.data.Dataset):
    def __init__(self, data_root='./data/YouHome-multilabels/',
                 image_transform=None, is_multi_labels=False, known_labels=0, training=False, testing=False, val=False, cross_cam_mode=0):
        print("call YouHomeDataset Init")
        print(data_root)
        self.data_root = data_root
        self.img_names = []

        print(f"is_multi_labels set to be {is_multi_labels}")
        if is_multi_labels:
            self.num_labels = 73 # for all the labels
        else: 
            self.num_labels = 45 # for just activities labels 
        self.testing = testing
        self.labels = []
        self.img_dir = data_root
        # print(training, testing, val)
        if training:
            self.img_dir = self.data_root + "train/image"
            self.labels_path = self.data_root + "train/label"
        elif testing:
            self.img_dir = self.data_root + "test/image"
            self.labels_path = self.data_root + "test/label"
        elif val:
            self.img_dir = self.data_root + "val/image"
            self.labels_path = self.data_root + "val/label"
            
        for name in glob(os.path.join(self.img_dir, '*.jpg')):
            img_name = name.split("/")[-1][:-4]
            self.img_names.append(img_name)
            label_file = os.path.join(self.labels_path, img_name+ '.json')
            label_vector = np.zeros(self.num_labels)

            #print(f"open file {img_name}")
            
            with open(label_file) as f:
                data = json.load(f)
            for activity in data['activities']:
                if len(activity.strip()) and ("None" not in activity):
                    activity = activity.strip()
                    label_vector[int(category_info[activity])] = 1.0
            if is_multi_labels:
                if len(data['posture'].strip()):
                    label_vector[int(category_info[data['posture']])] = 1.0
                if len(data['ID'].strip()):
                    label_vector[int(category_info[data['ID']])] = 1.0
                if len(data['gender'].strip()):
                    label_vector[int(category_info[data['gender']])] = 1.0

            self.labels.append(label_vector)
        self.labels = np.array(self.labels).astype(int)
        #print(self.labels) 
        self.image_transform = image_transform
        self.epoch = 1
    def __getitem__(self, index):
        name = self.img_names[index] + '.jpg'
        image = Image.open(os.path.join(self.img_dir, name)).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        labels = torch.Tensor(self.labels[index])
        sample = {}
        sample['image'] = image
        sample['labels'] = labels

        return sample

    def __len__(self):
        return len(self.img_names)


# train_dataset = YouHomeDataset()