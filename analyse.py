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
    df.loc[(df.index.hour >= 19) | (df.index.hour <= 5), 'pred'] = df.true.min()
    df.loc[(df.index.hour >= 19) | (df.index.hour <= 5), 'pred0'] = df.true0.min()
    df2 = df[df.index.month>0]
    return df2
root_path = r"/kaggle/working/S-D-Mamba/results2/"
root_path = r"C:\scholar\results2/"
#root_path = r"C:\scholar\results (3)\kaggle\working\S-D-Mamba\results2/"

l = os.listdir(root_path)
path_list = []
for i in range(0,len(l)):
    path_list.append(root_path+l[i]+r"/result.csv")

df_list = []
for i in path_list:
    df_list.append(postprecess(i))


for i in range(len(l)):
    df = df_list[i]
    print("r2",l[i],r2_score(df.true,df.pred))
    print("mse",l[i],mean_squared_error(df.true0,df.pred0))

start = 10000
gap = 500
c = df.iloc[start:start+gap]
plt.figure(figsize=[15,5])

for i in range(len(l)):
    c = df_list[i].iloc[start:start+gap]
    plt.plot(c.index,c.pred,label = l[i])

plt.plot(c.index,c.true,label = "groundtruth")

plt.legend()
plt.show()

result_list = []
for i in range(len(l)):
    df = df_list[i]
    model = l[i]
    monthly_data = []
    for name, group in df.groupby(pd.Grouper(freq='M')):
        row = {}
        if i == 0:
            row["month"] = name.month
            row["count"] = group.shape[0]
        monthly_r2 = r2_score(group.true, group.pred)
        monthly_mse = mean_squared_error(group.true0, group.pred0)
        row['R2_{}'.format(model)] = monthly_r2
        row['MSE_{}'.format(model)] = monthly_mse
        monthly_data.append(row)

    result_df = pd.DataFrame(monthly_data)
    result_list.append(result_df)
result = pd.concat(result_list, axis=1)
cols = list(result.columns)
cols.sort(reverse = True)
result = result[cols]
print(result)