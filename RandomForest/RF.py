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

# command line inputs
# input_fd = '../data/cleaned2'
# output_fd = './cleaned2_first8'
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


sele_cols = list(filter(lambda x: ('new_group'+'_' in x) or ('X314' in x) or ('X8_' in x) or ('X2_' in x) , TRAIN.columns))
TRAIN = TRAIN[sele_cols]
TEST= TEST[sele_cols]


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
RF
----------------------""" 
Nfeature = TRAIN.shape[1]
rf_fit = ensemble.RandomForestRegressor(n_estimators = 800, max_features = Nfeature//2, verbose=1,n_jobs=2,\
                                        oob_score=True,max_depth=5)
rf_fit.fit(TRAIN,y)
rf_fit.oob_score_

# report results on training data
imp = rf_fit.feature_importances_
imp = pd.Series(imp,index=TRAIN.columns)
imp.sort_values(ascending=False,inplace=True)
Nshow =50
imp = imp.iloc[:Nshow]

f= plt.figure(figsize=(5,18))
plt.barh(range(0,len(imp)),imp.values[::-1],align='center')
plt.yticks(range(0,len(imp)),imp.index[::-1])
f.savefig(os.path.join(output_fd,'imp.pdf'))

g = open(os.path.join(output_fd,'report.txt'),'w')
g.write('OOB R2: '+str(rf_fit.oob_score_) )
g.close()

"""----------------------
GENERATE PREDICTIONS
----------------------""" 
pred =rf_fit.predict(TEST)
pred = pd.DataFrame(pred,index = TEST.index,columns=['y'])
pred.to_csv(output_fd+'/RF_0702.csv',index_label='ID')
