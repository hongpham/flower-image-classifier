#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Hong Pham

import argparse
from Classifier import Classifier
from PIL import Image
import numpy as np
import json
import torch
from torch import optim
import numpy as np

'''
This script predict the species of flowers based on an image. It loads a pretrained model saved in checkpoint.pth file, then use this model for prediction.  

usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  data_dir checkpoint

Predict flower name from an input image

positional arguments:
  data_dir              Location of the image
  checkpoint            Location of checkpoint.pth file

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Top K most likely flower classes
  --category_names CATEGORY_NAMES
                        File mapping of categories to real names
  --gpu                 Use GPU for training
'''

def get_input_args():
    '''
    Retrieve input arguments from user, then return args object
    '''
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Predict flower name from an input image")
    parser.add_argument("data_dir", help="Location of the image", type=str)
    parser.add_argument("checkpoint", help="Location of checkpoint.pth file", type=str)
    parser.add_argument("--top_k", help="Top K most likely flower classes", \
                        default=1, type=int)
    parser.add_argument("--category_names", help="File mapping of categories to real names", type=str)
    parser.add_argument("--gpu", help = "Use GPU for training", action="store_true")
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    '''
    Create a new model based on checkpoint file of pretrained model. Return a new model 
    '''
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = Classifier(checkpoint['in_units'],
                       checkpoint['hidden_layers_1'],
                       checkpoint['hidden_layers_2'],
                       checkpoint['out_units'])

    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #resize imgage
    img = Image.open(image)
    img.thumbnail((256,256))
    
    #crop image at center
    width,height = img.size
    upper = (height/2) - (224/2)
    lower = (height/2) + (224/2)
    left = (width/2)  - (224/2)
    right = (width/2) + (224/2)
    img = img.crop((left, upper, right, lower))
    #convert color channels from int to float
    color_arr = np.array(img)
    float_color_arr = (color_arr/255).astype(float)
    #normalized image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_color = (float_color_arr - mean)/std
    
    #reorder dimension and return
    processed_image = np.transpose(normalized_color,(2,0,1))
    
    return processed_image

def predict(image, model, k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()
  
    # change image from array to tensor
    tensor_image = torch.from_numpy(image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device)
    #run prediction, retrieve top 5 prob and index
    output = model(tensor_image)
    prob = torch.exp(output)
    top_prob, top_idx = prob.topk(k, dim=1)

    #convert top_idx tensor to array
    top_prob, top_idx = top_prob.to('cpu'), top_idx.to('cpu')
    top_prob, top_idx = top_prob.detach().numpy(), top_idx.numpy()
    top_prob_arr = np.around(top_prob[0].astype(float), decimals = 2)
    top_idx_arr = top_idx[0]
    
    class_to_idx = model.class_to_idx
    top_class = []
    
    for i in top_idx_arr:
        for cls,idx in class_to_idx.items():
            if (i == idx):
                top_class.append(cls)
                
    return top_prob_arr, top_class

def print_result(k, top_prob_arr, top_class, catagory_names):
    top_prob = []
    for p in top_prob_arr:
        top_prob.append("{:.1%}".format(p))

    if catagory_names is None:
        if k == 1:
            print('Class: {0:10} \t  Probability: {1}'.format(top_class[0],top_prob[0]))
        elif k <= 0:
            print('top_k value need to be a possitive interger')
        else:
            for i in range(k):
                print('Class: {0:10} \t  Probability: {1}'.format(top_class[i],top_prob[i]))
    else:
        with open(catagory_names, 'r') as f:
            cat_to_name = json.load(f)        
        
        name = []
        for cls in top_class:
            if cls in cat_to_name:
                name.append(cat_to_name[cls])
                
        if k == 1:
            print('Class: {:>10} \t Name: {:>10} \t Probability: {:>10}'.format(top_class[0],name[0],top_prob[0]))
        elif k <= 0:
            print('top_k value need to be a possitive interger')
        else:
            for i in range(k):
                print('Class: {:>12} \t Name: {:>12} \t Probability: {:>12}'.format(top_class[i],name[i],top_prob[i]))
                
def main():
    
    arg_parse = get_input_args()
    data_dir = arg_parse.data_dir
    checkpoint = arg_parse.checkpoint
    top_k = arg_parse.top_k
    catagory_names = arg_parse.category_names
    gpu = arg_parse.gpu
    
    #load_checkpoint, process image, and retrieve probability
    model = load_checkpoint(checkpoint)
    image = process_image(data_dir)
    top_prob, top_class = predict(image,model,top_k,gpu)
    
    print_result(top_k, top_prob, top_class, catagory_names)
    
             
if __name__ == "__main__":
    main()
       