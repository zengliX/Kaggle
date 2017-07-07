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
import mca
from sklearn.decomposition import PCA, FastICA

# command line inputs
# input_fd = '../data/cleaned3'
# output_fd = './temp'
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

ncomp=5

# MCA
mca_out = mca.mca(temp,ncols=ncomp)
mca_mat = pd.DataFrame(mca_out.fs_r_sup(temp,ncomp),index=temp.index)


# ICA
ica = FastICA(n_components=ncomp, random_state=420)
ica_mat =pd.DataFrame(ica.fit_transform(temp),index=temp.index)


# append
for i in range(min(ncomp,mca_mat.shape[1])):
    TRAIN['mca_' + str(i)] = mca_mat.loc[TRAIN.index,i]
    TEST['mca_' + str(i)] = mca_mat.loc[TEST.index,i]

for i in range(min(ncomp,ica_mat.shape[1])):
    TRAIN['ica_' + str(i)] = ica_mat.loc[TRAIN.index, i]
    TEST['ica_' + str(i)] = ica_mat.loc[TEST.index, i]


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
params_list = param_gen(eta=[.05,.01,.005],max_depth=[2,3,4],sub=[1,.8,.9],col_sub=[.3,.6,1])

#params_list = param_gen(eta=[.01],max_depth=[2,3,4],sub=[1],col_sub=[1])

"""----------------------
CROSS VALIDATION IN TRAINING
----------------------""" 
np.random.seed(23)

"""
xgb_params = {
    'eta': 0.005,
    'max_depth': 2,
    'subsample': 0.93,
#    'colsample_bytree':0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1,
    'base_score': 0
}
"""

# binary columns

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
        y_train, y_test = res.iloc[train_ind], res.iloc[test_ind]
        
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
        out['train_r2'].append(r2_score( y.iloc[train_ind], y_linear_train[train_ind] + model.predict(dtrain)))
        out['test_r2'].append(r2_score(y.iloc[test_ind], y_linear_train[test_ind]+model.predict(dtest)))
        ct += 1
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out

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

# remove highest and lowest
#test_r2 = [x['fit']['test_r2'] for x in cv_out]
#def temp_f(x):
#    return np.sort(x)[1:-1].mean()
#new_mean = list(map(temp_f,test_r2))
#final_params = cv_out[np.argmax(new_mean)]['pars']

final_params = cv_out[np.argmax([x['fit']['test_r2_mean'] for x in cv_out])]['pars']


"""----------------------
FINAL MODEL
----------------------""" 

# form DMatrices for xgb training
dtrain = xgb.DMatrix(TRAIN, res, feature_names=TRAIN.columns.values)
dtest = xgb.DMatrix(TEST)


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
output.loc[[289,624,5816,6585,7420,7805],:] = 100.63 # set to mean
output.to_csv(output_fd+'/XGB_withLinear_tuned.csv',index_label='ID')


fig, ax = plt.subplots(figsize=(12,30))
xgb.plot_importance(model2,height=0.8, ax=ax)
fig.savefig(output_fd+'/imp.pdf')


