#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Hong Pham
import argparse

def get_input_args():

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint")
    parser.add_argument("data_dir", help="Directory of dataset", type=str)
    parser.add_argument("--save_dir", help="Directory to save checkpoints. \
                        Default is current directory", default='.',type=str)
    parser.add_argument("--arch", help="Choose CNN Architecture between densenet121 and vgg16", \
                        default='vgg16', choices=['vgg16', 'densenet121'], type=str)
    parser.add_argument("--learning_rate", help="Set learning rate", default=0.0001, type=float)
    parser.add_argument("--hidden_units", help = "Number units for last hidden layers.\
                        Must be between 1000-103 for vgg16 and 256-103 for densenet121. Default is 1000 for vgg16 ", default=1000 ,type=int)
    parser.add_argument("--epochs", help = "Number of epochs. Default is 25", default=25, type = int)
    parser.add_argument("--gpu", help = "Use GPU for training", action="store_true")
    
    args = parser.parse_args()

    return args
