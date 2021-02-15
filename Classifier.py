#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Author: Hong Pham

import torch
import torch.nn.functional as F
from torch import nn
'''
Create a simple NN architecture 
'''
class Classifier(nn.Module):
    
    def __init__(self,in_units, hidden_layer_1, hidden_layer_2, out_units):
        super().__init__()
        self.fc1 = nn.Linear(in_units, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, out_units)
        
        #dropout with 50 percent
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)      
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x 