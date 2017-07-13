"""
Averaging results from different methods
author: li zeng
"""
import pandas as pd
import numpy as np

# cd into the averaging folder

file_lst = ['top_out/new_xgb_tuned_p49.csv',\
            'top_out/tune2_p49.csv',\
            'top_out/XGB_linear_tuned_p49.csv',\
            'top_out/LGB_withLinear_mcaica_randtuned.csv',\
            'top_out/LGB2000.csv']
ct=0
fs=['rank1','rank3','rank2','LGB1000','LGB2000']
for f in file_lst:
    temp = pd.DataFrame.from_csv(f)
    temp.columns=[fs[ct]]
    #temp.columns=['y'+f]
    if not 'out' in globals():
        out = temp
    else:
        out= pd.concat([out,temp],axis=1)
    ct += 1

print(out.corr())

out.loc[[289,624,5816,6585,7420,7805],:]

ave = out.apply(np.mean,1)
ave.to_frame(name='y').to_csv('aveof5.csv',index_label='ID')
