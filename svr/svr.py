"""
Random Forest implementation
author: li zeng
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn import ensemble
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import itertools

# command line inputs
# input_fd = '../data/cleaned3'
# output_fd = './svr_all'
_, input_fd, output_fd = sys.argv

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

"""
sele_cols = list(filter(lambda x: ('new_group'+'_' in x) or ('X314' in x) \
                        or ('X3_' in x) or ('X4_' in x) or ('X0_' in x)\
                        , TRAIN.columns))
TRAIN = TRAIN[sele_cols]
TEST= TEST[sele_cols]
"""


"""
X0_cols = list(filter(lambda x: 'X0'+'_' in x, TRAIN.columns))
TRAIN.drop(X0_cols,axis=1,inplace=True)
TEST.drop(X0_cols,axis=1,inplace=True)
"""


"""----------------------
LINEAR MODEL with X0
----------------------""" 
X0_cols = list(filter(lambda x: 'X0'+'_' in x, TRAIN.columns))

linear_fit = LinearRegression(fit_intercept=False)
linear_fit.fit(TRAIN[X0_cols],y)
SCORE0  =linear_fit.score(TRAIN[X0_cols],y)        
y_linear_train = linear_fit.predict(TRAIN[X0_cols]) # linear pred on train
y_linear_test = linear_fit.predict(TEST[X0_cols]) # linear pred on test
res = y - y_linear_train # residual


# drop X0 columns
TRAIN.drop(X0_cols,axis=1,inplace=True)
TEST.drop(X0_cols,axis=1,inplace=True)


"""----------------------
SVR CV
----------------------""" 

# generate parameter
def param_gen(kernel,gammalist,Clist,eps):
    out= []
    for k,g,c,e in itertools.product(kernel,gammalist,Clist,eps):
        k0,k1 = k.split(' ')
        params = {
        'kernel': k0,
        'degree': int(k1),
        'gamma': g,
        'C': c,
        'epsilon': e,
        }
        out.append(params)
    return out

Nfeature = TRAIN.shape[1]
param_list = param_gen(kernel = ['rbf 0','poly 2','sigmoid 0'],gammalist=[0.5/Nfeature,1/Nfeature,5/Nfeature],\
                       Clist=[0.1,1,10],eps=[0.01,0.1,0.3])

p = param_list[-1]
# cross validation
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
        y_train, y_test = res.iloc[train_ind], res.iloc[test_ind]
        
        # fit svr
        svr_fit = SVR(kernel=p['kernel'],degree=p['degree'],gamma=p['gamma'],coef0=1,tol=0.00001,C=p['C'],epsilon=p['epsilon'])
        svr_fit.fit(X_train,y_train)
        pred_train = svr_fit.predict(X_train)
        plt.scatter(pred_train,y_train)
        svr_fit.score(X_train,y_train)        
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out
