# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:39:50 2019

@author: Ljx
"""


from .BasicModule import BasicModule

import torch as t, numpy as np
import torch.nn as nn
from .kpywavelet import wavelet as kpywavelet

from torch.autograd import Variable

class Wavelet(BasicModule):
    
    def __init__(self, opt):
        super(Wavelet, self).__init__()
        self.module_name = 'Wavelet'
        self.opt = opt
        
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        self.lstm_layer = 2
        self.cnn_out_channel = 1
        
        
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=self.input_size, out_channels=self.input_size*2, kernel_size=(3,3),bias = False),
                nn.BatchNorm2d(self.input_size*2),
                nn.Conv2d(in_channels=self.input_size*2, out_channels=self.input_size*4, kernel_size=(3,3), bias = False),
                nn.BatchNorm2d(self.input_size*4),
                nn.Conv2d(in_channels=self.input_size*4, out_channels=self.cnn_out_channel, kernel_size=(5,7), stride = (3,3), bias = False),
                )
        self.cnn_linear_rnn = nn.Linear(31*18, 4*32)
        self.encoder = nn.LSTM(4, self.encoder_hidden_size, self.lstm_layer)
        self.decoder = nn.LSTMCell(4, self.decoder_hidden_size)
        self.out_linear = nn.Linear(self.decoder_hidden_size + self.output_size, self.output_size)

    # input_data : T * batch_size * 1(input_size) 
    def forward(self, input_data):
        
        wavelet_power = self.WaveletTransform(input_data)
        cnn_out = self.cnn(wavelet_power).squeeze() # batch * 31 * 18
        cnn_out = self.cnn_linear_rnn(cnn_out.reshape(input_data.shape[1], -1))
        cnn_out = cnn_out.reshape(32, input_data.shape[1], 4)
        
        encoder_hidden = self.init_encoder_inner(cnn_out)
        encoder_cell = self.init_encoder_inner(cnn_out)
        # en_h_out : 1 * batch_size * encoder_hidden_size
        en_outs_h, (en_h_out, en_c_out) = self.encoder(cnn_out, (encoder_hidden, encoder_cell))
        
        context = cnn_out[-1]
        decoder_hidden = en_h_out[1]
        decoder_cell = en_c_out[1]
        # out_data : future * batch_size * output_size
        out_data = t.zeros(self.opt.future, context.size(0), self.output_size, dtype=t.float64).to(self.opt.device)
        for i in range(self.opt.future):
#            context = nn.functional.dropout(context, p = .5)
            decoder_hidden, decoder_cell = self.decoder(context, (decoder_hidden, decoder_cell))
            context = self.out_linear(t.cat((context, decoder_hidden), dim = 1))
            out_data[i, :, :] = context
        
        return out_data
        
        
        
    def init_encoder_inner(self, x):
        return Variable(x.data.new(self.lstm_layer, x.size(1), self.encoder_hidden_size).zero_())
        
        
    def WaveletTransform(self, input_data):
        data = (input_data - input_data.mean(dim = 0)) / input_data.std(dim = 0)
        dt = 1
        dj = 1./20
        mother = kpywavelet.Morlet(6.) # Morlet mother wavelet with wavenumber=6
        # data_ : batch * size     *      T
        data_ = data.reshape(data.size(0), data.size(1) * data.size(2)).permute(1,0).cpu().numpy()
        f_dim = int(np.log2(data.size(0) / 2) / dj) + 1
        t_dim = int(data.size(0))
        power = np.zeros([data.size(1) * data.size(2), f_dim, t_dim])
        for i in range(data_.shape[0]):
            wave, scales, freqs, coi, dj, s0, J = kpywavelet.cwt(data_[i, ...], dt, dj=dj, s0=-1, J=-1, wavelet=mother)
#            print(wave.shape)
#            print(power[0].shape)
            power[i] = np.power(wave, 2)
        power = t.from_numpy(power).reshape(data.size(1), data.size(2), f_dim, t_dim).to(self.opt.device)
        #power : batch * input_data_size * F.dim * T.dim
        return power
        
        