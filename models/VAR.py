# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:23:18 2019

@author: Ljx
"""


from .BasicModule import BasicModule
from .BasicVar import BasicVar

import torch as t
import torch.nn as nn
from torch.autograd import Variable

class VAR():
    
    def __init__(self, opt):
        super(VAR, self).__init__()
        self.module_name = 'VAR'
        self.opt = opt
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        

    # input_data : T * batch_size * 1(input_size) 
    def predict(self, input_target_data):
        input_data = input_target_data[0]
        var_predicted = t.zeros(input_data.size(1), self.opt.future, self.output_size, dtype=t.float64).to(self.opt.device)
        for batch in range(input_data.shape[1]):
            to_var = input_data[:, batch, :]
            var_predicted[batch, :, :] = BasicVar(to_var.detach().cpu().numpy(), h = self.opt.future, needLog = self.opt.needLog)
        var_predicted = var_predicted.permute(1, 0, 2)
        return var_predicted


if __name__ == '__main__':
    var = VAR()
    print(var.module_name)