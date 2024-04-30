from data_provider.data_factory import data_provider
from utils import Args
import matplotlib.pyplot as plt

arg = Args.Args(root_path=r"C:\Repo\pv_data\Data/")
flag = "train"
data_set, data_loader  = data_provider(arg, flag)

#for x,y,xt,yt,xtt,ytt in data_set:
for x, y, xt, yt in data_set:
    print(x.shape)
    a = y[:,-1]
    print(y.shape)

    print(xt.shape)
    tt = yt
    break
"""
b= data_set.inverse_transform(a)
ttt = data_set.data_stamp_reverse(tt)

plt.plot(ttt,b)
plt.show()
"""
