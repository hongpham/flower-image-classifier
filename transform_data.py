#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Hong Pham

import torch
from torchvision import datasets, transforms
'''
Load and transform image for train and validation. Image need to store in a directory following specific format as below:

    data_dir is the path as input provided to this script 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
   
'''
def transform_data(image_path):
    data_dir = image_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    #define transform
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(50),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    train_data = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle=True)

    return train_data, valid_data, train_datasets.class_to_idx