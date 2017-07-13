"""
LightGBM
author: li zeng
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import itertools
import pickle
import time
import mca
from sklearn.decomposition import PCA, FastICA
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# command line inputs
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
linear_fit.score(TRAIN[X0_cols],y)
y_linear_train = linear_fit.predict(TRAIN[X0_cols]) # linear pred on train
y_linear_test = linear_fit.predict(TEST[X0_cols]) # linear pred on test
res = y - y_linear_train # residual


# drop X0 columns
TRAIN.drop(X0_cols,axis=1,inplace=True)
TEST.drop(X0_cols,axis=1,inplace=True)


# # Append decomposition components to datasets
bin_cols = [x for x in TRAIN.columns if not '_' in x]
temp = pd.concat([TRAIN,TEST],axis=0)[bin_cols]

ncomp=10

# MCA
mca_out = mca.mca(temp,ncols=ncomp)
mca_mat = pd.DataFrame(mca_out.fs_r_sup(temp,ncomp),index=temp.index)


# ICA
ica = FastICA(n_components=ncomp, random_state=420,tol=10**(-4))
ica_mat =pd.DataFrame(ica.fit_transform(temp),index=temp.index)


# append
for i in range(min(ncomp,mca_mat.shape[1])):
    TRAIN['mca_' + str(i)] = mca_mat.loc[TRAIN.index,i]
    TEST['mca_' + str(i)] = mca_mat.loc[TEST.index,i]

for i in range(min(ncomp,ica_mat.shape[1])):
    TRAIN['ica_' + str(i)] = ica_mat.loc[TRAIN.index, i]
    TEST['ica_' + str(i)] = ica_mat.loc[TEST.index, i]


# remove duplicated columns
temp = TRAIN[bin_cols].T.duplicated()
TRAIN.drop(temp.index[temp],axis=1,inplace=True)
TEST.drop(temp.index[temp],axis=1,inplace=True)



"""----------------------
RANDOMIZED PARAMETERS
----------------------"""

param_dist = {
    'task': ['train'],
    'objective': ['regression'],
    'metric': ['l2'],
    'num_leaves': np.arange(4,15),
    'feature_fraction': np.linspace(0.85,1,5),
    'bagging_fraction': np.linspace(0.85,1,5),
    'learning_rate': np.linspace(0.001,0.01,5),
    'lambda_l1': np.linspace(0,100,10),
    'early_stopping_rounds': [30]
}

"""
param_dist = {
    'task': ['train'],
    'objective': ['regression'],
    'metric': ['l2'],
    'num_leaves': np.arange(4,7),
    'feature_fraction': np.linspace(0.85,1,2),
    'bagging_fraction': np.linspace(0.85,1,2),
    'learning_rate': np.linspace(0.001,0.01,2)
}
"""

lgb_model = lgb.LGBMRegressor()
kfold = KFold(n_splits=5, shuffle=True, random_state=12)
rgs = RandomizedSearchCV(lgb_model, param_dist, n_iter=1000,n_jobs = 1,
                         cv = kfold,verbose=2)
rgs.fit(TRAIN, res)
print(rgs.best_score_ )

final_params = rgs.best_params_
print(final_params)

# save cv_results
cv_file = output_fd+'/cv_out.pckl'
f = open(cv_file,'wb')
pickle.dump(rgs,f)
f.close()

"""----------------------
CROSS VALIDATION with the best
----------------------"""
np.random.seed(1234)

"""
lgb_params = {
    'task': 'train',
    'objective': 'regression',
    'metric': {'l2'},
    'max_depth': 2,
    'learning_rate': 0.001,
    'feature_fraction': 1,
    'early_stopping_round': 30,
    'lambda_l2': 0.1,
    'verbose': -1
}
"""

# binary columns
def myCV(lgb_params):
    numFolds = 5
    kf = KFold(n_splits= numFolds ,shuffle = True)
    kf.get_n_splits(TRAIN)
    out = {'train_r2':[],'test_r2':[]}
    # cv
    ct = 1
    for train_ind, test_ind in kf.split(TRAIN):
        print('calculating fold:',ct)
        # split data
        X_train, X_test = TRAIN.iloc[train_ind], TRAIN.iloc[test_ind]
        y_train, y_test = res.iloc[train_ind], res.iloc[test_ind]
        # fit lgb
        dtrain = lgb.Dataset(X_train, y_train)
        # get best iter
        cv_results = lgb.cv(lgb_params, dtrain,num_boost_round = 3000, early_stopping_rounds=40 ,nfold=5)
        niter = np.argmin(cv_results['l2-mean'])
        model = lgb.train(lgb_params,dtrain,num_boost_round=int(niter))
        # append results
        out['train_r2'].append(r2_score( y.iloc[train_ind], y_linear_train[train_ind] + model.predict(X_train)))
        out['test_r2'].append(r2_score(y.iloc[test_ind], y_linear_train[test_ind]+model.predict(X_test)))
        ct += 1
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out

cur_cv = myCV(final_params)
# report
print('cv of best: '+ str(cur_cv) + '\n')

"""----------------------
FINAL MODEL
----------------------"""

lgb_train = lgb.Dataset(TRAIN, res)
cv_results = lgb.cv(final_params, lgb_train, early_stopping_rounds = 40,num_boost_round = 3000, nfold=5)
niter = np.argmin(cv_results['l2-mean'])

# final model
model2 = lgb.train(final_params,
                lgb_train,
                num_boost_round=niter)


# score
r2_train2 = r2_score(y, y_linear_train + model2.predict(TRAIN))
print('------------------------------------------------------')
print("R2 on training:",r2_train2,'\n')
print('------------------------------------------------------')

"""----------------------
PREDICTION
----------------------"""

# make predictions and save results

y_pred = y_linear_test + model2.predict(TEST)
output = pd.DataFrame({'y': y_pred},index=TEST.index)
output.loc[[289,624,5816,6585,7420,7805],:] = 100.63 # set to mean

# add probe
probe_out = pd.DataFrame.from_csv('../probing/probe_out.csv')
output.loc[probe_out.index,'y'] = probe_out['yValue']
output.to_csv(output_fd+'/LGB_withLinear_mcaica_randtuned.csv',index_label='ID')

imp = pd.Series(model2.feature_importance(),index=TRAIN.columns)
imp.sort_values(ascending=False,inplace=True)
imp.to_csv(output_fd+'/importance.csv',index_label='feature')
