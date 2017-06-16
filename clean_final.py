"""
final data preprocessing script
author: li zeng
"""

import pandas as pd
import numpy as np

# tr = train.copy(deep=True)
def preproc(tr,test=None,percent_cut = 0.02):
    """
    percent_cut: select variables that both group frequency > percent_cut
    """
    # change categorical to dummy variables
    tr = pd.get_dummies(tr,columns = tr.select_dtypes(['object']).columns)
    
    # remove columns with low variance
    to_del = []
    for col in tr.columns:
        if tr[col].dtype == 'int64':
            zero_freq = np.sum(tr[col]==0)/len(tr)
            if zero_freq > 0.98 or zero_freq < 0.02:
                to_del.append(col)
    tr = tr.drop(to_del,axis=1)
    
    
    return tr