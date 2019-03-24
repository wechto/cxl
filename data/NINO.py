# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:53:20 2018

@author: Ljx
"""

import os
from torch.utils import data
import numpy as np, pandas as pd
import torch as t


class NINO(data.Dataset):
    
    def __init__(self, o):
#        print(os.getcwd())
        dir_ = 'data/nino_data/'
#        dir_ = 'nino_data/'
        files = ['Nino1-2 1950-2017', 'Nino3 1950-2017', 'Nino3-4 1950-2017', 'Nino4 1950-2017']
        def getD(file, t_ = False):
            a = pd.read_csv(dir_ + file, header = None)
            b = np.array(a)
            c = np.array([float(b[i][0].split(' ')[-2 if t_ else -1]) for i in range(b.shape[0])])
            return c
        a = getD(files[0])
        b = getD(files[1])
        c = getD(files[2])
        d = getD(files[3])
        ts = getD(files[0], t_ = True)
    
        data = t.from_numpy(np.vstack([a, b, c, d]).T)
        ts = t.from_numpy(ts)
        
        N = ts.shape[0]
        
        self.T, self.future, self.tr_va_te = o.T, o.future, o.tr_va_te
            
        p1, p2 = 0.78, 0.8
        if o.T > 64:
            p1, p2 = 0.6, 0.7
        
        move = o.T + o.future - 1
        if self.tr_va_te == 0:
            self.data = data[ : int(N * p1) + move]
            self.ts = ts[ : int(N * p1) + move]
        if self.tr_va_te == 1:
            self.data = data[int(N * p1) : int(N * p2) + move]
            self.ts = ts[int(N * p1) : int(N * p2) + move]
        if self.tr_va_te == 2:
            self.data = data[int(N * p2) : ]
            self.ts = ts[int(N * p2) : ]
                
    def __getitem__(self, index):
        return (self.data[index:index+self.T].clone().detach(), \
                self.data[index + self.T:index + self.T + self.future].clone().detach()), \
                (self.ts[index:index+self.T].clone().detach(), \
                 self.ts[index + self.T:index + self.T + self.future].clone().detach())
        
    def __len__(self):
        return self.data.size()[0] - self.T - self.future
    
    
if __name__ == '__main__':
    print(os.getcwd())
    class O():
        T, future = 200, 5
        tr_va_te = 2
        
    o = O()
    nino = NINO(o)
    print(nino.__len__())
    nino_data, nino_ts = nino[0]
    data = nino_data[0][:,0].numpy()
    
    import numpy as np
    from scipy.fftpack import fft,ifft
    import matplotlib.pyplot as plt
    import seaborn
    y = data
    x = np.linspace(0, 1, len(y))
    
    ylimax = 0.3
    plt_f_lim = int(len(x)/2)
    ffty = fft(y)/len(x)
    plt.subplot(3,2,1);
    plt.plot(x, y)
#    plt.show()
    plt.subplot(3,2,2);
    plt.plot(x[0:plt_f_lim], ffty[0:plt_f_lim]);plt.ylim([0,ylimax])
    plt.show()
    ffty_ = ffty
    ffty_[abs(ffty_)<0.3] = 0
    iffty_ = ifft(ffty_)
    plt.subplot(3,2,5);
    plt.plot(x, iffty_)
#    plt.show()
    plt.subplot(3,2,6);
    plt.plot(x[0:plt_f_lim], ffty_[0:plt_f_lim]);plt.ylim([0,ylimax])
    plt.show()
    

    
    
    

    