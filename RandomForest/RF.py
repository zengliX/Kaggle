"""
Random Forest implementation
author: li zeng
"""
import os
os.chdir(os.path.realpath(os.curdir))
import sys
sys.path.append(os.path.realpath(os.curdir)+'/..')
import pandas as pd
import numpy as np
from sklearn import ensemble
from matplotlib import pyplot as plt

# command line inputs
# input_fd = '../data/raw'
# output_fd = './raw'
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
RF
----------------------""" 
np.random.seed(23)
Nfeature = TRAIN.shape[1]
rf_fit = ensemble.RandomForestRegressor(n_estimators = 800, max_features = Nfeature//2, verbose=1,n_jobs=2,\
                                        oob_score=True)
rf_fit.fit(TRAIN,y)

# report results on training data
imp = rf_fit.feature_importances_
imp = pd.Series(imp,index=TRAIN.columns)
imp.sort_values(ascending=False,inplace=True)
Nshow =30
imp = imp.iloc[:Nshow]

f = plt.figure()
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
pred.to_csv(output_fd+'/RF_0617.csv',index_label='ID')
