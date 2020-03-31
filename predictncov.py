# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:42:03 2020

@author: Chen Chen

Prediction of confirm cases of coronavirus using local linear models
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

def ll(y=cases,x=dates,predict_date=predict_date,h=3.35,weight=1):
    '''
    Local linear estimation is a way to regress a 
    random variable y given random variable x in a non-parametric way. 
    It works for both uni-variate and multi-variate data. 
    It includes automatic bandwidth determination. 
    
    y,x: array_like
    Datapoints to estimate from. Currently, it only supports 1-D array.
    
    bw_method:str, scalar or callable, optional
    The method used to calculate the estimator bandwidth. 
    This can be ‘scott’, ‘silverman’, a scalar constant. 
    If a scalar, this will be used directly as kde.factor. 
    If None (default), ‘scott’ is used. See Notes for more details.
    
    weights:array_like, optional
    weights of datapoints. This must be the same shape as dataset. 
    If None (default), the samples are assumed to be equally weighted
    
    The rows of matrix are points to fit, and the columns are the points to predict.
    As we are cross validating, the matrix xx and so on are square matrices.
    ''' 
  
    x = x.reshape((-1,1))    
    y = y.reshape((-1,1)) 
    
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


prediction = ll(cases,dates)

import matplotlib.pyplot as plt
plt.plot(dates, cases)
plt.plot(predict_date,prediction,'o')
plt.show()

