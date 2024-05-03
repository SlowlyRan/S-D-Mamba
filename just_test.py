import pandas as pd
"""df_raw = pd.read_csv("C:\Repo\pv_data\Data\pv_data.csv")
print(df_raw.shape)
"""
import numpy as np
ll=[]
for i in range(5):
    new = np.ones((1,3))
    print(new)
    ll.append(new)
ll = np.concatenate(ll,axis=0)

print(ll)