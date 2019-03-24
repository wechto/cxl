# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:52:29 2018

@author: Ljx
"""
import torch as t
import time
import os

class BasicModule(t.nn.Module):
    
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))
        
    def load(self, path):
        self.load_state_dict(t.load(path))
        
    def save(self, opt, name = None):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if name is None:
            prefix = 'checkpoints/' + opt.model + '.'+opt.data +'.'
            name = time.strftime(prefix + '%m%d%H%M%S.pth')
        t.save(self.state_dict(), name)
        return name

if __name__ == '__main__':
    bm = BasicModule()
    print(bm.module_name)