from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import os
import matplotlib.pyplot as plt


def postprecess(path):
    df = pd.read_csv(path)
    df.date = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    df.loc[(df.index.hour >= 20) | (df.index.hour <= 6), 'pred'] = df.true.min()
    df.loc[(df.index.hour >= 20) | (df.index.hour <= 6), 'pred0'] = df.true0.min()
    df2 = df[df.index.month>0]
    return df2

root_path = r"C:\scholar\result/"

l = os.listdir(root_path)
path_list = []
for i in range(0,len(l)):
    path_list.append(root_path+l[i]+r"\result.csv")