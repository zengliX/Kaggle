"""
xgboost
author: li zeng
"""
import os
os.chdir(os.path.realpath(os.curdir))
import sys
sys.path.append(os.path.realpath(os.curdir)+'/..')
import xgboost 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# command line inputs
#input_fd = '../data/cleaned'
#output_fd = './xgboost_cleaned'
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



"""----------------------
FIT MODEL
----------------------""" 

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}

dtrain = xgboost.DMatrix(TRAIN, y, feature_names=TRAIN.columns.values)
model = xgboost.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)


fig, ax = plt.subplots(figsize=(12,18))
xgboost.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
fig.savefig(output_fd+'/imp.pdf')


"""----------------------
PREDICTION
----------------------""" 
dtest = xgboost.DMatrix(TEST)
ypred = model.predict(dtest,ntree_limit=model.best_ntree_limit)
pd.DataFrame(ypred,index=TEST.index,columns=['y']).to_csv(output_fd+'/XGB_0621.csv',index_label='ID')
