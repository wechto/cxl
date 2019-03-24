# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:16:27 2018

@author: Ljx
"""

import pandas as pd, numpy as np, torch as t
import matplotlib.pyplot as plt
import pyflux as pf


def BasicVar(data, h = 5, needLog = True):
    N = data.shape[0]
    alpha = [chr(i) for i in range(97,123)]
    columns = alpha[:data.shape[1]]
    
    index = pd.date_range(start='1/1/2001', periods = N)
    
    data = pd.DataFrame(np.array(data), index = index, columns = columns)
    #data = data.append(data,  ignore_index=True)
    data_log = np.log(data) if needLog else data
    #print('data:\n', data)
    #print('data_log:\n', data_log)
    
    #plt.figure(figsize=(15,5));
    #plt.plot(data.index,data);
    #plt.legend(data.columns.values,loc=3);
    #plt.title("Logged data");
    
    from statsmodels.tsa.stattools import adfuller
    def test_stationarity(timeseries):
        dftest = adfuller(timeseries, autolag='AIC')
        return dftest[1]
    
#    for i in columns:
#        print('p:', test_stationarity(np.array(data_log[i])))
        
    import statsmodels.tsa.stattools as st
#    lags = 0
#    for i in columns:
#        order = st.arma_order_select_ic(np.array(data_log[i]),max_ar=3,max_ma=3,ic=['aic', 'bic', 'hqic'])
#        lags += order.bic_min_order[0]
#    
#    lags = int(lags/4)
#    print('lags:',lags)
    lags = 2
    
    model = pf.VAR(data = data_log, lags=lags, integ=1)
    model.fit()
    
#    h = 1
#    print('h:', h)
    data_log_diff_predicted = model.predict(h = h)
    data_log_diff_predicted.columns = columns
    #print('data_log_diff_predicted:\n', data_log_diff_predicted)
    
    index = data.shape[0] - data_log_diff_predicted.shape[0]
    index = 5
    
    data_log_predicted = pd.DataFrame(data_log[data_log.index[-1]:]).append(data_log_diff_predicted).cumsum()
    #print('data_log_predicted:\n', data_log_predicted)
    data_predicted = np.exp(data_log_predicted)  if needLog else data_log_predicted 
    #print('data_predicted:\n', data_predicted)
    return t.from_numpy(np.array(data_predicted)[1:])

    
if __name__ == '__main__':
    
    class O():
        T, future = 20, 5
        tr_va_te = 0
        
    o = O()
    nino = NINO(o)
    print(nino.__len__())
    nino_data, nino_ts = nino[0]
    data_predicted = BasicVar(nino_data[0])
    print(data_predicted)
