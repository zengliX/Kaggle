"""
xgb with selection
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
from sklearn import ensemble
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


X0_cols = list(filter(lambda x: 'X0'+'_' in x, TRAIN.columns))
TRAIN.drop(X0_cols,axis=1,inplace=True)
TEST.drop(X0_cols,axis=1,inplace=True)


"""
# only use X1 - X8 + new group
keep = ['X1','X2','X3','X4','X5','X6','X8','new_group']
exp_keep = []
for k in keep:
    exp_keep += list(filter(lambda x: k+'_' in x, TRAIN.columns))

len(exp_keep)

TRAIN = TRAIN.loc[:,exp_keep]
TEST = TEST.loc[:,exp_keep]
"""

"""----------------------
GENERATE PARAMETERS
----------------------""" 

def param_gen(eta,max_depth,sub,col_sub):
    out= []
    for e,m,s,cs in itertools.product(eta,max_depth,sub,col_sub):
        xgb_params = {
        'eta': e,
        'max_depth': m,
        'subsample': s,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'colsample_bytree':cs,
        'silent': 1}
        out.append(xgb_params)
    return out

#params_list = param_gen(ntree=[500,400,600],eta=[.05,.02,.1],max_depth=[2,3,4,5,6],sub=[1,.8,.9])
params_list = param_gen(eta=[.05,.01,.005],max_depth=[2,3,4,5],sub=[1,.8,.9],col_sub=[.3,.6,1])

#params_list = param_gen(ntree=[500],eta=[0.1,.05],max_depth=[2,3],sub=[1])

"""----------------------
CROSS VALIDATION IN TRAINING
----------------------""" 
np.random.seed(1)

"""
xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
"""


# CV function
def myCV(xgb_params):
    numFolds = 10
    kf = KFold(n_splits= numFolds ,shuffle = True)
    kf.get_n_splits(TRAIN)

    out = {'train_r2':[],'test_r2':[]}
    ct = 1
    for train_ind, test_ind in kf.split(TRAIN):
        print('calculating fold:',ct)
        # split data
        X_train, X_test = TRAIN.iloc[train_ind], TRAIN.iloc[test_ind]
        y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
        
        # random forest variable selection
        Nfeature = X_train.shape[1]
        rf_fit = ensemble.RandomForestRegressor(n_estimators = 500, max_features = Nfeature//2, verbose=1,n_jobs=2,\
                                        oob_score=True)
        rf_fit.fit(X_train,y_train)
        imp = rf_fit.feature_importances_
        imp = pd.Series(imp,index=TRAIN.columns)
        imp.sort_values(ascending=False,inplace=True)
        Nselect = 150
        imp = imp.iloc[:Nselect]
        
        X_train = X_train[imp.index]
        X_test = X_test[imp.index]
        
        # fit xgboost
        dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns.values)
        dtest = xgb.DMatrix(X_test,y_test)
        cv_result = xgb.cv(xgb_params, 
                           dtrain, 
                           num_boost_round=3000, # increase to have better results (~700)
                           early_stopping_rounds=40,
                           verbose_eval=50, 
                           show_stdv=False
                           )
        niter = cv_result.shape[0]
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=niter)
        out['train_r2'].append(r2_score(dtrain.get_label(), model.predict(dtrain)))
        out['test_r2'].append(r2_score(dtest.get_label(), model.predict(dtest)))
        ct += 1
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out


# CV procedure
ct= 0
cv_out = []
t0 = time.time()
for xgb_params in params_list:
    print('------------------------------------------------------\n')
    print("working on parameters:",xgb_params,'\n')
    print("index:",ct,'\n')
    cur_cv = myCV(xgb_params)
    cv_out.append({'pars':xgb_params,'fit':cur_cv})
    ct+=1
    
    # report
    print("current cv results:",cur_cv)
    speed = (time.time()-t0)/ct
    print('time remaining:',(len(params_list)-ct)*speed/60,'mins\n')
    print('------------------------------------------------------')
    
# save cv_results
cv_file = output_fd+'/cv_out.pckl'
f = open(cv_file,'wb')
pickle.dump(cv_out,f)
f.close()
print('------------------------------------------------------')
print('mean-test-r2:',[x['fit']['test_r2_mean'] for x in cv_out])
print('------------------------------------------------------')

# final parameter
final_params = cv_out[np.argmax([x['fit']['test_r2_mean'] for x in cv_out])]['pars']


"""----------------------
FINAL MODEL
----------------------""" 

# form DMatrices for xgb training

Nfeature = TRAIN.shape[1]
rf_fit = ensemble.RandomForestRegressor(n_estimators = 500, max_features = Nfeature//2, verbose=1,n_jobs=2,\
                                        oob_score=True)
rf_fit.fit(TRAIN,y)
imp = rf_fit.feature_importances_
imp = pd.Series(imp,index=TRAIN.columns)
imp.sort_values(ascending=False,inplace=True)
Nselect = 150
imp = imp.iloc[:Nselect]
        
X_train = TRAIN[imp.index]
X_test = TEST[imp.index]


dtrain = xgb.DMatrix(X_train, y, feature_names=X_train.columns.values)
dtest = xgb.DMatrix(X_test)


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
model2 = xgb.train(dict(final_params, silent=0), dtrain, num_boost_round=niter)

# score
r2_train2 = r2_score(dtrain.get_label(), model2.predict(dtrain))
print("R2 on training:",r2_train2,'\n')


"""----------------------
PREDICTION
----------------------""" 

# make predictions and save results
# y_pred = model1.predict(dtest)
# output = pd.DataFrame({'ID': TEST.index, 'y': y_pred})
# output.to_csv(output_fd+'/XGB_tuned1.csv',index=False)

y_pred = model2.predict(dtest)
output = pd.DataFrame({'ID': TEST.index, 'y': y_pred})
output.to_csv(output_fd+'/XGB_tuned2.csv',index=False)


fig, ax = plt.subplots(figsize=(12,30))
xgb.plot_importance(model2,height=0.8, ax=ax)
fig.savefig(output_fd+'/imp.pdf')


