"""
Averaging results from different methods
author: li zeng
"""
import pandas as pd
import numpy as np

# cd into the averaging folder

file_lst = ['../RandomForest/RF_benchmark/RF_0617.csv',\
'../xgboost/xgboost_cleaned/XGB_0621.csv']

for f in file_lst:
    temp = pd.DataFrame.from_csv(f)
    #temp.columns=['y'+f]
    if not 'out' in globals():
        out = temp
    else:
        out= pd.concat([out,temp],axis=1)

ave = out.apply(np.mean,1)
ave.to_frame(name='y').to_csv('ave_0622.csv',index_label='ID')
