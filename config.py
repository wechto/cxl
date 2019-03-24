# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 09:52:15 2019

@author: Ljx
"""

import warnings
import torch as t


class DefaultConfig(object):
    
    env = 'default'
    model = 'LSTM'
    data = 'NINO'
    
    data_root = None
    load_model_path = None
    
    input_size, output_size = None, None
    needLog = None
    tr_va_te = None
    encoder_hidden_size = 64
    decoder_hidden_size = 64
    
    PID_history_length = 128
    
    batch_size = 8 # 7, 3
    T, future = 64, 4
    use_gpu = True
    print_freq = 20
    
    max_epoch = 1000
    lr = 0.01 # initial learning rate
    lr_decay = .95 # when val_loss incress, lr = lr * lr_dacay
    weight_decay = 1e-5
    
    num = 80 # for test & val

    
    device = t.device('cuda:0') if use_gpu else t.device('cpu')
    
    _input_kv = {'NINO':4}
    _output_kv = {'NINO':4}
    _needLog_kv = {'NINO':True}
    _data_max = {'NINO':30}
    
    def _parse(self, kwargs = {}, printconfig = False):
        '''
        更新参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribute %s' % k)
            setattr(self, k, v)
        if printconfig:
            print('\nuser config:')
            for k, v in self.__class__.__dict__.items():
                if not k.startswith('_'):
                    print(k, getattr(self, k))
        

opt = DefaultConfig()

if __name__ == '__main__':
    opt._parse(printconfig = True)
    print(opt.lr)

    
    
