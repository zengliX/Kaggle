"""
compare results for submission
author: li zeng
"""

from sys import argv
import pandas as pd
import numpy as np

#f1 = 'RandomForest/RF_Linear_step/RF_0704.csv'
#f2 = 'xgboost/xgb_cleaned2_all_tuned0630/XGB_tuned2.csv'

f1,f2 =  argv[1:]

y1 = pd.DataFrame.from_csv(f1)
y2 = pd.DataFrame.from_csv(f2)

y2 = y2.loc[y1.index]

print((y1 - y2).describe())
print(np.corrcoef(y1.y,y2.y))

#y2[(np.abs(y1-y2)>40).iloc[:,0]]
#y1[(np.abs(y1-y2)>40).iloc[:,0]]
