# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:43:01 2019

@author: Ljx
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:36:41 2019

@author: Ljx
"""


from .BasicModule import BasicModule

import torch as t
import torch.nn as nn
from torch.autograd import Variable

class GRU(BasicModule):
    
    def __init__(self, opt):
        super(GRU, self).__init__()
        self.module_name = 'GRU'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        print('input_size:', self.input_size, 'output_size:', self.output_size)
        
        self.encoder = nn.GRU(self.input_size, self.encoder_hidden_size, 1)
        self.decoder_in = nn.Linear(self.encoder_hidden_size, self.output_size)
        self.decoder = nn.GRUCell(self.input_size , self.decoder_hidden_size)
        self.out_linear = nn.Linear(self.decoder_hidden_size + self.output_size, self.output_size)

    # input_data : T * batch_size * 1(input_size) 
    def forward(self, input_data):
        encoder_hidden = self.init_encoder_inner(input_data)
        # en_h_out : 1 * batch_size * encoder_hidden_size
        en_outs_h, en_h_out = self.encoder(input_data, encoder_hidden)
        
        context = input_data[-1]
        decoder_hidden = en_h_out.squeeze(0)

        # out_data : future * batch_size * output_size
        out_data = t.zeros(self.opt.future, context.size(0), self.output_size, dtype=t.float64).to(self.opt.device)
        for i in range(self.opt.future):
#            context = nn.functional.dropout(context, p = .5)
            decoder_hidden = self.decoder(context, decoder_hidden)
            context = self.out_linear(t.cat((context, decoder_hidden), dim = 1))
            out_data[i, :, :] = context
        
        return out_data
        
    def init_encoder_inner(self, x):
        return Variable(x.data.new(1, x.size(1), self.encoder_hidden_size).zero_())
    
    def init_decoder_inner(self, x):
        return Variable(x.data.new(x.size(0), self.decoder_hidden_size).zero_())


if __name__ == '__main__':
    gru = GRU()
    print(gru.module_name)