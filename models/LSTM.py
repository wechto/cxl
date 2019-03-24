# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:04:50 2019

@author: Ljx
"""

from .BasicModule import BasicModule

import torch as t
import torch.nn as nn
from torch.autograd import Variable

class LSTM(BasicModule):
    
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.module_name = 'LSTM'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        print('input_size:', self.input_size, 'output_size:', self.output_size)
        
        self.encoder = nn.LSTM(self.input_size, self.encoder_hidden_size, 1)
        self.decoder_in = nn.Linear(self.encoder_hidden_size, self.output_size)
        self.decoder = nn.LSTMCell(self.input_size , self.decoder_hidden_size)
        self.out_linear = nn.Linear(self.decoder_hidden_size + self.output_size, self.output_size)

    # input_data : T * batch_size * 1(input_size) 
    def forward(self, input_data):
        encoder_hidden = self.init_encoder_inner(input_data)
        encoder_cell = self.init_encoder_inner(input_data)
        # en_h_out : 1 * batch_size * encoder_hidden_size
        en_outs_h, (en_h_out, en_c_out) = self.encoder(input_data, (encoder_hidden, encoder_cell))
        
        context = input_data[-1]
        decoder_hidden = en_h_out.squeeze(0)
        decoder_cell = en_c_out.squeeze(0)
        # out_data : future * batch_size * output_size
        out_data = t.zeros(self.opt.future, context.size(0), self.output_size, dtype=t.float64).to(self.opt.device)
        for i in range(self.opt.future):
#            context = nn.functional.dropout(context, p = .5)
            decoder_hidden, decoder_cell = self.decoder(context, (decoder_hidden, decoder_cell))
            context = self.out_linear(t.cat((context, decoder_hidden), dim = 1))
            out_data[i, :, :] = context
        
        return out_data
        
    def init_encoder_inner(self, x):
        return Variable(x.data.new(1, x.size(1), self.encoder_hidden_size).zero_())
    
    def init_decoder_inner(self, x):
        return Variable(x.data.new(x.size(0), self.decoder_hidden_size).zero_())


if __name__ == '__main__':
    lstm = LSTM()
    print(lstm.module_name)