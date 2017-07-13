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
import time
import pickle
# command line inputs
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

# select columns
keep =['X47','X5','X127','X267','X383','X1','X351','X240','X8','X51','X152','X104','X241','X163','X19','X132','X345']
def f(x):
    return any([(z == x) or (z+'_' in x) for z in keep ])
tests_cols = [x for x in TRAIN.columns if f(x)]

TRAIN = TRAIN[tests_cols]
TEST = TEST[tests_cols]

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
                       Clist=np.linspace(0.1,3,10),eps=np.linspace(0.01,0.1,3))
#param_list = param_gen(kernel = ['rbf 0'],gammalist=[0.5/Nfeature],Clist=[0.1],eps=[0.1,0.2])


p = param_list[-1]
# cross validation
def myCV(p):
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
        pred_test = svr_fit.predict(X_test)
        out['train_r2'].append(r2_score(y.iloc[train_ind],pred_train + y_linear_train[train_ind]))
        out['test_r2'].append(r2_score(y.iloc[test_ind],pred_test + y_linear_train[test_ind]))
        ct += 1
    out['train_r2_mean']=np.mean(out['train_r2'])
    out['test_r2_mean']=np.mean(out['test_r2'])
    return out

# loop through param_list
ct= 0
cv_out = []
t0 = time.time()
for p in param_list:
    print('------------------------------------------------------\n')
    print("working on parameters:",p,'\n')
    print("index:",ct,'\n')
    cur_cv = myCV(p)
    cv_out.append({'pars':p,'fit':cur_cv})
    ct+=1

    # report
    print("current cv results:",cur_cv)
    speed = (time.time()-t0)/ct
    print('time remaining:',(len(param_list)-ct)*speed/60,'mins\n')
    print('------------------------------------------------------')

    if ct%5==0:
        temp = [x['fit']['test_r2_mean'] for x in cv_out]
        print('Best test_r2_mean so far:',max(temp))

# save cv_results
cv_file = output_fd+'/cv_out.pckl'
f = open(cv_file,'wb')
pickle.dump(cv_out,f)
f.close()
print('------------------------------------------------------')
print('mean-test-r2:',[x['fit']['test_r2_mean'] for x in cv_out])
print('------------------------------------------------------')


"""----------------------
FINAL MODEL
----------------------"""
