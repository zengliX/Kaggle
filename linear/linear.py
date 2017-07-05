"""
linear model
author: li zeng
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import pickle
import time

# command line inputs
#input_fd = '../data/cleaned2'
#output_fd = './temp'
input_fd=sys.argv[1] 
output_fd = sys.argv[2]

if not os.path.exists(output_fd):
    os.makedirs(output_fd)
    
    
"""----------------------
LOAD DATA
----------------------""" 
TRAIN = pd.DataFrame.from_csv(os.path.join(input_fd,'train.csv'))
TEST = pd.DataFrame.from_csv(os.path.join(input_fd,'test.csv'))

y = TRAIN.y
del TRAIN['y']
del TEST['y']


sele_cols = list(filter(lambda x: ('X0'+'_' in x), TRAIN.columns))
#X314 = list(filter(lambda x: '314'in x, TRAIN.columns))
TRAIN = TRAIN[sele_cols]
TEST= TEST[sele_cols]


"""----------------------
CROSS VALIDATION IN TRAINING
----------------------""" 
np.random.seed(23)


def myCV():
    numFolds = 5
    kf = KFold(n_splits= numFolds ,shuffle = True)
    kf.get_n_splits(TRAIN)

    out = {'train_r2':[],'test_r2':[]}
    ct = 1
    for train_ind, test_ind in kf.split(TRAIN):
        print('calculating fold:',ct)
        # split data
        X_train, X_test = TRAIN.iloc[train_ind], TRAIN.iloc[test_ind]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
    
        # fit xgboost
        linear_fit = LinearRegression(fit_intercept=True)
        linear_fit.fit(X_train,y_train)
        #print(linear_fit.score(X_train,y_train))
        #print(linear_fit.score(X_test,y_test))
        out['train_r2'].append(linear_fit.score(X_train,y_train))
        out['test_r2'].append(linear_fit.score(X_test,y_test))
        ct += 1
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out

myCV()

"""----------------------
PREDICTION
----------------------""" 
linear_fit = LinearRegression(fit_intercept=True)
linear_fit.fit(TRAIN,y)
linear_fit.score(TRAIN,y)        
pred = linear_fit.predict(TEST)
pd.DataFrame(pred,columns=['y'],index=TEST.index).to_csv(output_fd+'/linear_X0.csv',index_label='ID')
