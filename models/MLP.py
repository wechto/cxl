# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:03:47 2019

@author: Ljx
"""


from .BasicModule import BasicModule

import torch as t
import torch.nn as nn
from torch.autograd import Variable

class MLP(BasicModule):
    
    def __init__(self, opt):
        super(MLP, self).__init__()
        self.module_name = 'MLP'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size

        print('input_size:', self.input_size, 'output_size:', self.output_size)
        
        self.linear1 = nn.Linear(int(self.opt.T * self.input_size / 1), int(self.opt.T * self.input_size / 2))
        self.linear2 = nn.Linear(int(self.opt.T * self.input_size / 2), int(self.opt.T * self.input_size / 4))
        self.linear3 = nn.Linear(int(self.opt.T * self.input_size / 4), int(self.opt.T * self.input_size / 8))
        self.linear4 = nn.Linear(int(self.opt.T * self.input_size / 8), int(self.opt.future * self.output_size))

    # input_data : T * batch_size * 1(input_size) 
    def forward(self, input_data):
        batch = input_data.shape[1]
        x = input_data.reshape(batch, -1)
        x = self.linear1(x)
        x = nn.functional.dropout(x, p = .2)
        x = self.linear2(x)
        x = nn.functional.dropout(x, p = .2)
        x = self.linear3(x)
        x = nn.functional.dropout(x, p = .2)
        x = self.linear4(x)
        out_data = x.reshape(self.opt.future, batch, self.output_size)
        return out_data
        


if __name__ == '__main__':
    mlp = MLP()
    print(mlp.module_name)