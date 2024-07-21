import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2
from pdb import set_trace as stop
import os, random

from youhome_dataset import YouHomeDataset
import warnings

warnings.filterwarnings("ignore")
# This is the data from 20K data training set.
precomputed_activity_dict = {
    0: 313,
    1: 350,
    2: 1450,
    3: 607,
    4: 150,
    5: 1841,
    6: 20,       #original value is 9
    7: 585,
    8: 36,
    9: 199,
    10: 540,
    11: 183,
    12: 3824,
    13: 351,
    14: 1637,
    15: 59,
    16: 322,
    17: 1050,
    18: 20,      # original value is 9
    19: 21,
    20: 16,
    21: 23,
    22: 104,
    23: 20,   # original value is 5
    24: 35,
    25: 103,
    26: 70,
    27: 20,  # original value is 8
    28: 72,
    29: 60,
    30: 69,
    31: 162,
    32: 58,
    33: 125,
    34: 150,
    35: 165,
    36: 71,
    37: 111,
    38: 645,
    39: 20,   # original value is 14
    40: 20,   # original value is 6
    41: 20,
    42: 20,   # original value is 1
    43: 149,
    44: 222,
}

class NoOpTransform:
    def __call__(self, img):
        return img
    
def getYouHomeSampler(dataset):
    activity_dict = precomputed_activity_dict
    
    # for idx, sample in enumerate(dataset):
    #     #print(sample['labels'][0].item())
    #     activity = int(sample['labels'][0].item())
    #     activity_dict[activity] = activity_dict.get(activity, 0) + 1
    
    sorted_dict = dict(sorted(activity_dict.items(), key=lambda item: item[0]))
    print(f"the raw inputs activity_dict = {sorted_dict}")
    
    sample_weights = [0] * len(dataset)

    for idx, sample in enumerate(dataset):
        activity = int(sample['labels'][0].item())
        #activity = sample['labels'][0]
        if activity_dict[activity] > 0:
            sample_weights[idx] = 1.0 / activity_dict[activity]

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler

def get_data(args):
    dataset = args.dataset
    data_root = args.dataroot
    batch_size = args.batch_size
    run_testing_only = args.run_testing_only

    rescale = args.scale_size
    random_crop = args.crop_size
    workers = args.workers
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = rescale
    crop_size = random_crop
    if args.test_batch_size == -1:
        args.test_batch_size = batch_size
    if args.dataset == 'youhome_activity':
        # no normalization for single activity youhome dataset (at least for color images)
        trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.RandomChoice([
                                                NoOpTransform(),
                                                transforms.RandomCrop(640),
                                                transforms.RandomCrop(576),
                                                transforms.RandomCrop(512),
                                                transforms.RandomCrop(384),
                                                #transforms.RandomCrop(320)
                                            ]),
                                            transforms.Resize((crop_size, crop_size)),
                                            transforms.RandomChoice([
                                                NoOpTransform(),
                                                #transforms.Lambda(lambda x: x),  # No change to the original image
                                                transforms.RandomHorizontalFlip(),
                                                NoOpTransform(),
                                                transforms.RandomRotation(30),
                                            ]),
                                            # transforms.RandomChoice([
                                            #     NoOpTransform(),
                                            #     NoOpTransform(),
                                            #     #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                            #     transforms.v2.Grayscale(num_output_channels=3),
                                            # ]),
                                            transforms.ToTensor(),
                                            normTransform])
        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform])
    else:
        trainTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.RandomChoice([
                                                transforms.RandomCrop(640),
                                                transforms.RandomCrop(576),
                                                transforms.RandomCrop(512),
                                                transforms.RandomCrop(384),
                                                transforms.RandomCrop(320)
                                            ]),
                                            transforms.Resize((crop_size, crop_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normTransform])
        testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            normTransform])

    train_dataset = None
    train_loader = None
    test_dataset = None
    test_loader = None
    drop_last = False
    if dataset == 'youhome_multi' or dataset == 'youhome_activity':
        youhome_root = data_root #Your Dataset
        print("Start to load dataset")
        print("data_root = " + youhome_root)

        if run_testing_only == False:
            train_dataset = YouHomeDataset(
                data_root=youhome_root,
                image_transform=trainTransform,
                training=True)
            print("Training set loaded")
        # valid_dataset = YouHomeDataset(
        #     data_root=youhome_root,
        #     image_transform=testTransform,
        #     val=True)
        # print("Val set loaded")
        test_dataset = YouHomeDataset(
            data_root=youhome_root,
            image_transform=testTransform,
            known_labels=0,
            testing=True)
        print("Testing set loaded")
    else:
        print('no dataset avail')
        exit(0)

    if train_dataset is not None:
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, #shuffle=True, 
        #                           num_workers=workers,
        #                           sampler=getYouHomeSampler(train_dataset),
        #                           drop_last=drop_last)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=workers,
                                  drop_last=drop_last)
    # if valid_dataset is not None:
    #     valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)

    #return train_loader, valid_loader, test_loader
    return train_loader, test_loader
