# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:42:03 2020

@author: Chen Chen

Prediction of confirm cases of coronavirus using local linear models

Example of function inputs as numpy array are

dates
 
array([ 0.,  2.,  4., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47.,
       48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60.,
       61., 62., 63., 64., 65., 66., 67., 68.])
    
cases

array([   1,    3,    4,    6,    9,   15,   22,   25,   28,   36,   39,
         47,   61,   65,   78,   92,  112,  134,  171,  210,  267,  307,
        353,  436,  533,  669,  818, 1029, 1219, 1405, 1617, 1791, 1918,
       2032], dtype=int64)   

OR use lists as input 
"""

import pandas as pd
import numpy as np


data = pd.read_json(path_or_buf=('https://gist.githubusercontent.com/wileam/'
                                 '49d4c83efb6b554eb9ce773289dcc5f9/raw/edaa80'
                                 '199265af0278525cd46b8c33eec47aa2b4/'
                                 'NSWDailyHistorys.json'), orient='records')

#hyperparameter from cross validation
bandwidth=3.35

# dates
dates = (data['date'] - data['date'][0]).astype('timedelta64[D]').to_numpy()

#use cumulative cases if you want to predict cumulative.
#use daily confirmed cases if you want to predict daily confirmed.
cases = data['totalConfirmedNumber'].to_numpy()


#predict two day forward. The case number if changing quickly.
#Statistics shows that extrapolating anything more than 2 days is not reliable.
predict_date = np.hstack((dates,np.arange(1,3,1)+max(dates)))

def local_linear(y=cases,x=dates,N_days_predict=2,h=3.35,weight=1):
    '''
    Local linear estimation is a way to regress a 
    random variable y given random variable x in a non-parametric way. 
    It works for both uni-variate and multi-variate data. 
    It includes automatic bandwidth determination. 
    
    y,x: array_like
    Datapoints to estimate from. Currently, it only supports 1-D array.
    
    bw_method: scalar , optional
    This will be used directly as kde.factor. 
    If None (default), 3.35 is used, which is optimised for cumulative case
    numbers using cross validation.
    
    weights:array_like, optional
    weights of datapoints. This must be the same shape as dataset. 
    If None (default), the samples are assumed to be equally weighted
    
    The rows of matrix are points to fit, and the columns are the points to predict.
    As we are cross validating, the matrix xx and so on are square matrices.
    Do not change N_days_predict to anything >2 for now.
    '''  
    
    predict_date=np.hstack((dates,np.arange(1,1+N_days_predict,1)+max(dates)))
    
    x = np.asarray(x).reshape((-1,1))    
    y = np.asarray(y).reshape((-1,1)) 
    
    xx = np.absolute(x-predict_date.T)  

    
    K = np.copy(xx)
    xxh = np.copy(xx)/h     
    K[xxh<1] = 70/81*(1-xxh[xxh<1]**3)**3
    K[xxh>=1] = 0
    
    
    s0 = (K*weight).sum(0,keepdims =True)
    s1 = (K*xx*weight).sum(0,keepdims =True)
    s2 = (K*xx**2*weight).sum(0,keepdims =True)
    
    w = (s2-s1*xx*weight)*K/(s2*s0-s1*s1)  
    y_predict = (w*y).sum(0,keepdims =True)     
    y_predict = y_predict.T

    return y_predict.astype(int)


prediction = local_linear(cases,dates)

import matplotlib.pyplot as plt
plt.plot(dates, cases)
plt.plot(predict_date,prediction,'o')
plt.show()

