# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:45:29 2019

@author: Ljx
"""


from .BasicModule import BasicModule

import torch as t
import torch.nn as nn


class CNN(BasicModule):
    def __init__(self, opt):      
        super(CNN, self).__init__()
        self.module_name = 'CNN'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        
        
        self.cnn1 = nn.Conv2d(1, 4, [5,1], 1)
        self.cnn2 = nn.Conv2d(4, 8, [13,1], 1)
        self.cnn3 = nn.Conv2d(8, 32, [17, 4], 1)
        self.cnn4 = nn.Conv2d(1, 1, [3,3], 1)
        self.linear = nn.Linear(900, 16)
        
    
    def forward(self, input_data):
        batch = input_data.shape[1]
        x = input_data.reshape(batch, 1, self.opt.T, self.input_size)
        x = self.cnn3(self.cnn2(self.cnn1(x)))
        x = x.reshape(batch, 1, 32, 32)
        x = self.cnn4(x).reshape(batch, -1)
        out = self.linear(x).reshape(4, batch, 4)
        return out
        
    
    
if __name__ == '__main__':
    cnn = CNN()
    print(cnn.module_name)