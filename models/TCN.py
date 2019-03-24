# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:47:09 2019

@author: Ljx
"""


from .BasicModule import BasicModule

import torch as t
import torch.nn as nn


class TCN(BasicModule):
    def __init__(self, opt):      
        super(TCN, self).__init__()
        self.module_name = 'TCN'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        
        self.input_T = 76
        
        self.in_ = [1, 1, 1]
        self.out_ = [1, 1, 1]
        self.k_ = [(3,1), (3,1), (3,1)]
        self.s_ = [(1,1), (1,1), (1,1)]
        self.d_ = [(1,1), (2,1), (4,1)]
        
        self.input_T_transform = nn.Linear(self.opt.T, self.input_T)
        self.conv0 = ResidualBlock(self.in_, self.out_, self.k_, self.s_, self.d_)
        self.conv1 = ResidualBlock(self.in_, self.out_, self.k_, self.s_, self.d_)
        self.conv2 = ResidualBlock(self.in_, self.out_, self.k_, self.s_, self.d_)
        self.out_linear = nn.Linear((self.input_T - (2+4+8)*3) * self.input_size, self.output_size * self.opt.future)
        
    
    def forward(self, input_data):
        input_T_transformed = self.input_T_transform(input_data.permute(1, 2, 0).reshape(-1, self.opt.T))
        conv0_out = self.conv0(input_T_transformed.reshape(input_data.size(1), self.input_size, self.input_T).permute(0,2,1).unsqueeze(1))
        conv1_out = self.conv1(conv0_out)
        conv2_out = self.conv2(conv1_out)
        relu_out = nn.ReLU()(conv2_out)
        out = self.out_linear(relu_out.squeeze(1).reshape(input_data.size(1), -1))
        out = out.reshape(input_data.size(1), self.opt.future, self.output_size).permute(1, 0, 2)
        return out
        
        
        
class ResidualBlock(t.nn.Module):
    
    def __init__(self, in_, out_, k_, s_, d_, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.network = nn.Sequential(
                nn.Conv2d(in_channels=in_[0], out_channels=out_[0], kernel_size=k_[0], stride = s_[0], dilation = d_[0], bias = False),
                nn.BatchNorm2d(out_[0]),
                nn.Conv2d(in_channels=in_[1], out_channels=out_[1], kernel_size=k_[1], stride = s_[1], dilation = d_[1], bias = False),
                nn.BatchNorm2d(out_[1]),
                nn.Conv2d(in_channels=in_[2], out_channels=out_[2], kernel_size=k_[2], stride = s_[2], dilation = d_[2], bias = False),
                )
        self.highway = shortcut
        
    def forward(self, x):
        out = self.network(x)
        residual = x if self.highway is None else self.highway(x)
        out += residual[:, :, -out.shape[2]:,:]
        return t.nn.functional.relu(out)
    
    
if __name__ == '__main__':
    tcn = TCN()
    print(tcn.module_name)