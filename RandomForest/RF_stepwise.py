"""
Random Forest implementation
author: li zeng
"""
import os
import sys
sys.path.append(os.path.realpath(os.curdir)+'/..')
import pandas as pd
import numpy as np
from sklearn import ensemble
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import mca
from sklearn.decomposition import PCA, FastICA


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


# Append decomposition components to datasets
bin_cols = [x for x in TRAIN.columns if not '_' in x]
temp = pd.concat([TRAIN,TEST],axis=0)[bin_cols]

ncomp=15

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


# remove duplicated columns
temp = TRAIN[bin_cols].T.duplicated()
TRAIN.drop(temp.index[temp],axis=1,inplace=True)
TEST.drop(temp.index[temp],axis=1,inplace=True)





"""----------------------
RF step-wise selectino
----------------------"""
def onestep(cur_cols,pool):
    out=[]
    print('variables in pool:',pool)
    for z in pool:
        print('working on',z)
        temp = cur_cols + list(filter(lambda x: (z+'_' in x) or (z==x), TRAIN.columns))
        Nfeature = len(temp)
        rf_fit = ensemble.RandomForestRegressor(n_estimators = 800, max_features = max(Nfeature//2,1), verbose=0,n_jobs=2,\
                                        oob_score=True,max_depth=3)
        rf_fit.fit(TRAIN.loc[:,temp],res)
        out.append(rf_fit.oob_score_)
    return  out

def step_wise(keep):
    # step wise selection
    pool = keep.copy()
    cur_oob = -1
    new_oob = 0
    cur_cols = [x for x in TRAIN.columns if ('mca' in x) or ('ica'in x)]
    while pool:
        cur_oob = new_oob
        out = onestep(cur_cols,pool)
        print("max oob:",max(out))
        if max(out) < cur_oob: break
        loc = np.argmax(out)
        sele = pool.pop(loc)
        new_oob = out[loc]
        print('selecting:',sele)
        cur_cols += list(filter(lambda x: sele in x, TRAIN.columns))
    return (cur_cols,cur_oob)

keep =['X47','X5','X127','X267','X383','X1','X351','X240','X8','X51','X152','X104','X241','X163','X19','X132','X345']
# save model for each group
out = step_wise(keep)

"""----------------------
GENERATE PREDICTIONS
----------------------"""
Nfeature=len(out[0])
rf_fit = ensemble.RandomForestRegressor(n_estimators = 800, max_features = max(Nfeature//2,1), verbose=0,n_jobs=2,\
                                        oob_score=True,max_depth=3)
rf_fit.fit(TRAIN.loc[:,out[0]],res)
pred_train = y_linear_train + rf_fit.predict(TRAIN.loc[:,out[0]])
r2_score(y,pred_train)

pred_test = y_linear_test + rf_fit.predict(TEST.loc[:,out[0]])
pred_test = pd.DataFrame(pred_test,index= TEST.index,columns=['y'])
pred_test.loc[[289,624,5816,6585,7420,7805],:] = np.mean(y)

# add probe
probe_out = pd.DataFrame.from_csv('../probing/probe_out.csv')
pred_test.loc[probe_out.index,'y'] = probe_out['yValue']

pred_test.to_csv(output_fd+'/RF_step_res_p_0708.csv',index_label='ID')
