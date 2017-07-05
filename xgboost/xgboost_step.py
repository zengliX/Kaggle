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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import itertools
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


"""----------------------
LINEAR MODEL with X0
----------------------""" 
X0_cols = list(filter(lambda x: 'X0'+'_' in x, TRAIN.columns))

linear_fit = LinearRegression(fit_intercept=False)
linear_fit.fit(TRAIN[X0_cols],y)
cur_r2 = linear_fit.score(TRAIN[X0_cols],y)        
y_linear_train = linear_fit.predict(TRAIN[X0_cols]) # linear pred on train
y_linear_test = linear_fit.predict(TEST[X0_cols]) # linear pred on test
res = y - y_linear_train # residual


# drop X0 columns
TRAIN.drop(X0_cols,axis=1,inplace=True)
TEST.drop(X0_cols,axis=1,inplace=True)


"""----------------------
CROSS VALIDATION IN TRAINING
----------------------""" 
np.random.seed(23)


def myCV(xgb_params,mytrain):
    numFolds = 5
    kf = KFold(n_splits= numFolds ,shuffle = True)
    kf.get_n_splits(mytrain)

    out = {'train_r2':[],'test_r2':[]}
    ct = 1
    for train_ind, test_ind in kf.split(mytrain):
        print('calculating fold:',ct)
        # split data
        X_train, X_test = mytrain.iloc[train_ind], mytrain.iloc[test_ind]
        y_train, y_test = res.iloc[train_ind], res.iloc[test_ind]
    
        # fit xgboost
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
        dtest = xgb.DMatrix(X_test,y_test)
        cv_result = xgb.cv(xgb_params, 
                           dtrain, 
                           num_boost_round=3000, # increase to have better results (~700)
                           early_stopping_rounds=40,
                           verbose_eval=False, 
                           show_stdv=False
                           )
        niter = cv_result.shape[0]
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=niter)
        out['train_r2'].append(r2_score( y.iloc[train_ind], y_linear_train[train_ind] + model.predict(dtrain)))
        out['test_r2'].append(r2_score(y.iloc[test_ind], y_linear_train[test_ind]+model.predict(dtest)))
        ct += 1
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out

def onestep(params,cur_cols,pool):
    out =  np.zeros(len(pool))
    print('variables in pool:',pool)
    for i in range(len(pool)):
        z= pool[i]
        if (not cur_cols) and (not z in ['X1','X5','X8']): continue
        print('working on',z)
        temp = cur_cols + list(filter(lambda x: (z+'_' in x) or (z==x), TRAIN.columns))
        cur_cv = myCV(params,TRAIN[temp])
        out[i] = cur_cv['test_r2_mean']
    return  out

def step_wise(params,keep):
    # step wise selection
    pool = keep.copy()
    old_r2 = 0
    new_r2 = 0.60236
    cur_cols = []
    while pool:
        old_r2 = new_r2
        out = onestep(params,cur_cols,pool)
        print("max r2:",max(out))
        if max(out) < old_r2: break
        loc = np.argmax(out)
        sele = pool.pop(loc)
        new_r2 = out[loc]
        print('selecting:',sele)
        cur_cols += list(filter(lambda x: (sele+'_' in x) or (sele == x), TRAIN.columns))
    return (cur_cols,new_r2)



final_params = {'colsample_bytree': 0.3,
 'eta': 0.05,
 'eval_metric': 'rmse',
 'max_depth': 2,
 'objective': 'reg:linear',
 'silent': 1,
 'subsample': 1}

keep =['X47','X5','X127','X267','X383','X1','X351','X240','X8','X51','X152','X104','X241','X163','X19','X132','X345']

step_out = step_wise(final_params,keep)

myCV(final_params,TRAIN[step_out[0]])



"""----------------------
FINAL MODEL
----------------------""" 

# form DMatrices for xgb training
dtrain = xgb.DMatrix(TRAIN[step_out[0]], res, feature_names=step_out[0])
dtest = xgb.DMatrix(TEST[step_out[0]])


# xgb, cross-validation
cv_result = xgb.cv(final_params, 
                   dtrain, 
                   num_boost_round=3000, # increase to have better results (~700)
                   early_stopping_rounds=20,
                   verbose_eval=50, 
                   show_stdv=False
                  )
niter = cv_result.shape[0]

# train model
# model1 = xgb.train(dict(final_params, silent=0), dtrain, num_boost_round=niter_1sd)
model2 = xgb.train(dict(final_params, silent=0), dtrain, num_boost_round=niter)

# score
# r2_train1 = r2_score(dtrain.get_label(), model1.predict(dtrain))
# print("R2 on training:",r2_train1,'\n')
r2_train2 = r2_score(y, y_linear_train + model2.predict(dtrain))
print('------------------------------------------------------')
print("R2 on training:",r2_train2,'\n')
print('------------------------------------------------------')


"""----------------------
PREDICTION
----------------------""" 

# make predictions and save results
# y_pred = model1.predict(dtest)
# output = pd.DataFrame({'ID': TEST.index, 'y': y_pred})
# output.to_csv(output_fd+'/XGB_tuned1.csv',index=False)

y_pred = y_linear_test + model2.predict(dtest)

output = pd.DataFrame({'y': y_pred},index=TEST.index)
output.loc[[289,624,5816,6585,7420,7805],:] += 100.63 # set to mean
output.to_csv(output_fd+'/XGB_withLinear_step.csv',index_label='ID')


fig, ax = plt.subplots(figsize=(12,30))
xgb.plot_importance(model2,height=0.8, ax=ax)
fig.savefig(output_fd+'/imp.pdf')


