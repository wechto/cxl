# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:29:22 2018

@author: Ljx
"""
import warnings

from .NINO import NINO


class datasets(object):
    def __init__(self, opt):
        self.o = opt
        
    def getData(self):
        if self.o.data == 'NINO':
            return NINO(self.o)
        warnings.warn('Warning: opt has not attribute %s' % self.opt.data)
    