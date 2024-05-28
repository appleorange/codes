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

def getYouHomeSampler(dataset):
    activity_dict = {}
    for idx, sample in enumerate(dataset):
        #print(sample['labels'][0].item())
        activity = int(sample['labels'][0].item())
        activity_dict[activity] = activity_dict.get(activity, 0) + 1
    
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
                                                transforms.RandomCrop(640),
                                                transforms.RandomCrop(576),
                                                transforms.RandomCrop(512),
                                                transforms.RandomCrop(384),
                                                transforms.RandomCrop(320)
                                            ]),
                                            transforms.Resize((crop_size, crop_size)),
                                            transforms.RandomChoice([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip()
                                                #transforms.RandomRotation(90),
                                            ]),
                                            transforms.RandomChoice([
                                                transforms.v2.RandomPhotometricDistort(),
                                                transforms.v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
                                            ]),
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

    test_dataset = None
    test_loader = None
    drop_last = False
    if dataset == 'youhome_multi' or dataset == 'youhome_activity':
        youhome_root = data_root #Your Dataset
        print("Start to load dataset")
        print("data_root = " + youhome_root)
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, #shuffle=True, 
                                  num_workers=workers,
                                  sampler=getYouHomeSampler(train_dataset),
                                  drop_last=drop_last)
    # if valid_dataset is not None:
    #     valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)

    #return train_loader, valid_loader, test_loader
    return train_loader, test_loader
