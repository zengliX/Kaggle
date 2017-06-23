"""
xgb
author: li zeng
"""
import os
import sys
sys.path.append(os.path.realpath(os.curdir)+'/..')
import xgboost as xgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# command line inputs
#input_fd = '../data/cleaned2'
#output_fd = './xgb_cleaned2_first8'
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

X0_cols = list(filter(lambda x: 'X0'+'_' in x, TRAIN.columns))
TRAIN.drop(X0_cols,axis=1,inplace=True)
TEST.drop(X0_cols,axis=1,inplace=True)


# only use X1 - X8 + new group
keep = ['X1','X2','X3','X4','X5','X6','X8','new_group']
exp_keep = []
for k in keep:
    exp_keep += list(filter(lambda x: k+'_' in x, TRAIN.columns))

len(exp_keep)

TRAIN = TRAIN.loc[:,exp_keep]
TEST = TEST.loc[:,exp_keep]



"""----------------------
CROSS VALIDATION IN TRAINING
----------------------""" 



xgb_params = {
    'n_trees': 500, 
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


def myCV(xgb_params):
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
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
        dtest = xgb.DMatrix(X_test,y_test)
        cv_result = xgb.cv(xgb_params, 
                           dtrain, 
                           num_boost_round=1000, # increase to have better results (~700)
                           early_stopping_rounds=100,
                           verbose_eval=50, 
                           show_stdv=False
                           )
        niter = np.argmin(cv_result['test-rmse-mean']) # find best iteration
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=niter)
        out['train_r2'].append(r2_score(dtrain.get_label(), model.predict(dtrain)))
        out['test_r2'].append(r2_score(dtest.get_label(), model.predict(dtest)))
        ct += 1
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out

myCV(xgb_params)

"""----------------------
FINAL MODEL
----------------------""" 

xgb_params = {
    'n_trees': 500, 
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


# form DMatrices for xgb training
dtrain = xgb.DMatrix(TRAIN, y, feature_names=TRAIN.columns.values)
dtest = xgb.DMatrix(TEST)


# xgb, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1000, # increase to have better results (~700)
                   early_stopping_rounds=100,
                   verbose_eval=50, 
                   show_stdv=False
                  )

niter = np.argmin(cv_result['test-rmse-mean']) # find best iteration
        
# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=niter)

# score
r2_score(dtrain.get_label(), model.predict(dtrain))

"""----------------------
PREDICTION
----------------------""" 

# make predictions and save results
y_pred = model.predict(dtest)
output = pd.DataFrame({'ID': TEST.index, 'y': y_pred})
output.to_csv(output_fd+'/XGB_first8.csv')

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=80, height=0.8, ax=ax)
fig.savefig(output_fd+'/imp.pdf')


