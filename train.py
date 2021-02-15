#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Hong Pham

import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn, optim
from PIL import Image
import numpy as np
from workspace_utils import active_session
from get_input_args import get_input_args
from transform_data import transform_data
from Classifier import Classifier
  
'''
This script creates and trains a new model based on popular torch.models densenet121 or vgg16. It then save a checkpoint of this new model so that you can use this checkpoint for prediction or future model training.
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch {vgg16,densenet121}]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu]
                data_dir

Train a new network on a dataset and save the model as a checkpoint

positional arguments:
  data_dir              Directory of dataset

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory to save checkpoints. Default is current
                        directory
  --arch {vgg16,densenet121}
                        Choose CNN Architecture between densenet121 and vgg16
  --learning_rate LEARNING_RATE
                        Set learning rate
  --hidden_units HIDDEN_UNITS
                        Number units for last hidden layers. Must be between
                        1000-103 for vgg16 and 256-103 for densenet121.
                        Default is 1000 for vgg16
  --epochs EPOCHS       Number of epochs. Default is 25
  --gpu                 Use GPU for training
'''

def train_model(arch, hidden_layers_2, gpu, epochs, learning_rate, train_data, valid_data, save_dir, class_to_idx):
    '''
    Create a new model based on densenet121 or vgg16 specified by user input. 
    Train and validate this model while printing out loss and accuracy
    Once done, save the model as a checkpoint.pth file
    '''
    #parameters
    in_units = 0 
    out_units = 102 # pre_defined output units
    hidden_layers_1 = 0
    
    #choose model architecture 
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_units = 25088
        hidden_layers_1 = 4096

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_units = 1024
        hidden_layers_1 = 512
        
    #freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #define classifier
    model.classifier = Classifier(in_units, hidden_layers_1, hidden_layers_2, out_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)      

    #set model gpu
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'    
    model.to(device)

    print("Training  started ......")  
    with active_session():
        for e in range(epochs):
            train_loss = 0   
            for images,labels in train_data:
                images,labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                log_prob = model(images)
                print('finished training')
                loss = criterion(log_prob, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            else:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_data:
                        images,labels = images.to(device), labels.to(device)
                        log_prob = model(images)
                        valid_loss += criterion(log_prob,labels)
                        #get the prediction probability
                        prob = torch.exp(log_prob)
                        #get the most likely class and associated probability that network is predicting
                        top_prob, top_class = prob.topk(1, dim=1)
                        #assert if top class and labels is the same
                        assert_prediction = top_class == labels.view(*top_class.shape)
                        #calculate accuracy by taking average of assert_prediction
                        accuracy += torch.mean(assert_prediction.type(torch.FloatTensor))

                #Print validation loss and accuracy
                print("-----------------------")
                print("Epoch: {}/{}".format(e+1, epochs))
                print("Training Loss: {:.3f}".format(train_loss/len(train_data)))
                print("Validation Loss: {:.3f}".format(valid_loss/len(valid_data)))
                print("Validation Accuracy: {:.3f}".format(accuracy/len(valid_data)))

                model.train()
        
        #Saving checkpoint
        checkpoint_path = save_dir + '/checkpoint.pth'
        print("Saving checkpoint at {}".format(checkpoint_path))
        checkpoint = {'in_units': in_units,
                      'hidden_layers_1': hidden_layers_1,
                      'hidden_layers_2': hidden_layers_2,
                      'out_units': out_units,
                      'learning_rate': learning_rate,
                      'epochs': 25,
                      'model': model,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'class_to_idx': class_to_idx}
        torch.save(checkpoint, checkpoint_path)
        
def main():
    #get input arguments
    arg_parser = get_input_args()
    data_dir = arg_parser.data_dir
    save_dir = arg_parser.save_dir             
    network_arc = arg_parser.arch
    learning_rate = arg_parser.learning_rate
    hidden_units = arg_parser.hidden_units
    epochs = arg_parser.epochs
    gpu = arg_parser.gpu
    
    #retrieve and transform image data
    train_data, valid_data, class_to_idx = transform_data(data_dir)

    #train model and save checkpoint
    train_model(network_arc, hidden_units, gpu, epochs, learning_rate, train_data, valid_data, save_dir, class_to_idx)

# Call to main function to run the program
if __name__ == "__main__":
    main()
   